import torch.nn as nn

class BertCrossEncoder(nn.Module):
    def __init__(self, bert):
        self.model = bert
        self.linear_layer = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        bertrep, _ = self.model(input_ids, attention_mask=attention_mask)
        bertrep = bertrep[:, 0, :]
        output = self.linear_layer(bertrep)

        return output

class BertCandidateRanker(object):
    def __init__(self, cross_encoder, device="cpu", model_path=None, use_mlflow=False, logger=None):
        self.model = cross_encoder.to(device)
        self.device = device

        self.model_path = model_path
        self.use_mlflow = use_mlflow
        self.logger = logger

    def merge_mention_candidate(self, mention_id, candidate_ids):
        results = [mention_id + candidate_id[1:] for candidate_id in candidate_ids]
        return results

    def train(self,
              mention_dataset,
              candidate_dataset,
              inbatch=True,
              lr=1e-5,
              batch_size=32,
              random_bsz=100000,
              max_ctxt_len=32,
              max_title_len=50,
              max_desc_len=100,
              traindata_size=1000000,
              model_save_interval=10000,
              grad_acc_step=1,
              max_grad_norm=1.0,
              epochs=1,
              warmup_propotion=0.1,
              fp16=False,
              fp16_opt_level=None,
              parallel=False,
              hard_negative=False,
             ):


        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = get_scheduler(
            batch_size, grad_acc_step, epochs, warmup_propotion, optimizer, traindata_size)

        if fp16:
            assert fp16_opt_level is not None
            self.model, optimizer = to_fp16(self.model, optimizer, fp16_opt_level)

        if parallel:
            self.model = to_parallel(self.model)

        for e in range(epochs):
            dataloader = DataLoader(mention_dataset, batch_size=1, shuffle=True, collate_fn=my_collate_fn_json, num_workers=2)
            bar = tqdm(total=traindata_size)
            for step, (input_ids, labels, lines) in enumerate(dataloader):
                if self.logger:
                    self.logger.debug("%s step", step)
                    self.logger.debug("%s data in batch", len(input_ids))
                    self.logger.debug("%s unique labels in %s labels", len(set(labels)), len(labels))

                pages = labels
                for nn in lines[0]["nearest_neighbors"]:
                    if nn not in pages:
                        pages.append(str(nn))
                candidate_input_ids = candidate_dataset.get_pages(pages, max_title_len=max_title_len, max_desc_len=max_desc_len)

                inputs = self.merge_mention_candidate(input_ids[0], candidate_input_ids)

                inputs = pad_sequence([torch.LongTensor(token)
                                        for token in input], padding_value=0).t().to(self.device)
                input_mask = inputs > 0
                scores = self.model(inputs, input_mask)

                target = torch.LongTensor([0]).to(self.device)
                loss = F.cross_entropy(scores.unsqueeze(0), target, reduction="mean")

                if self.logger:
                    self.logger.debug("Accurac: %s", accuracy)
                    self.logger.debug("Train loss: %s", loss.item())


                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()


                if (step + 1) % grad_acc_step == 0:
                    if fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), max_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    if self.logger:
                        self.logger.debug("Back propagation in step %s", step+1)
                        self.logger.debug("LR: %s", scheduler.get_lr())

                if self.use_mlflow:
                    mlflow.log_metric("train loss", loss.item(), step=step)
                    mlflow.log_metric("accuracy", accuracy, step=step)

                if self.model_path is not None and step % model_save_interval == 0:
                    #torch.save(self.model.state_dict(), self.model_path)
                    save_model(self.model, self.model_path)

                bar.update(len(input_ids))
                bar.set_description(f"Loss: {loss.item()}, Accuracy: {accuracy}")

