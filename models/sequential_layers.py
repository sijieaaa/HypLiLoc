import torch.nn as nn


def conv(batch_norm, in_channels, out_channels, kernel_size=3, stride=1, dropout=0):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(0.1, inplace=True),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(dropout),
            # nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=0),
            # nn.MaxPool2d(2),
            # nn.LeakyReLU(0.2)
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2),
            # nn.LeakyReLU(0.1, inplace=True),
            # nn.Dropout(dropout),
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
            # nn.MaxPool2d(2),
            nn.LeakyReLU(0.1, inplace=True)
            # nn.ReLU()
            # nn.Dropout(dropout)
        )


def fc_dropout(hidden1, hidden2, p=0.2):
    return nn.Sequential(
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout2d(p)
    )
