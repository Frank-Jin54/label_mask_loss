import torch


class dynamic_encoding(torch.nn.Module):
    def __int__(self, num_labels):
        self.encode_transfer = torch.eye(num_labels, requires_grad=False)
        self.switched = set([_ for _ in range(num_labels)])
        self.num_labels = num_labels

    def calculate_loss(self, loss_fun, outputs, labels):
        # y: (batches, num_labels)
        # label: (batches)
        if self.num_labels:
            # still has some label need to be switched
            for y, label in zip(outputs, labels):
                switched_y_row = torch.argmax(y * self.encode_transfer)
                if switched_y_row != label:
                    # change label to y
                    if label in self.num_labels and switched_y_row in self.num_labels:
                        # swap the columns of matrix
                        idx = [_ for _ in range(self.num_labels)]
                        idx[switched_y_row], idx[label] = idx[label], idx[switched_y_row]
                        self.encode_transfer = self.encode_transfer.index_select(1, torch.LongTensor(idx))
                        self.num_labels.pop(label)

            return loss_fun(outputs * self.encode_transfer , labels)
        else:
            return loss_fun(outputs * self.encode_transfer, labels)
    def forward(self, y):
        # for inference
        return y * self.encode_transfer