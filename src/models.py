import torch
import torch.nn as nn
import torch.nn.functional as F

class TESSModelV5Aux(nn.Module):
    def __init__(self, n_aux_features = 1):
        super().__init__()
        
        self.aux_branch = nn.Sequential(
            nn.Linear(n_aux_features, 16),
            nn.ReLU(),
        )

        self.global_view = nn.Sequential(
            nn.Conv1d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = 61,
                stride = 4,
                #padding = 30
            ),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Dropout1d(p = 0.2),
            nn.MaxPool1d(kernel_size = 4),
            nn.Conv1d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = 31,
                stride = 2,
                padding = 15
            ),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout1d(p = 0.2),
            nn.MaxPool1d(kernel_size = 4)
        )

        self.focused_view = nn.Sequential(
            nn.Conv1d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = 5,
                stride = 2,
                #padding = 10
            ),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Dropout1d(p = 0.2),
            nn.MaxPool1d(kernel_size = 2),
            nn.Conv1d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = 5,
                stride = 1,
                #padding = 3
            ), 
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout1d(p = 0.2),
            nn.MaxPool1d(kernel_size = 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 128 + 16, out_features = 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(p = 0.5), 
            nn.Linear(in_features = 128, out_features = 1)
        )

    def pool_branch(self, feat):
        avg = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
        mn = -F.adaptive_max_pool1d(-feat, 1).squeeze(-1)
        return torch.cat([avg, mn], dim = 1)
        
    def forward(self, x, aux):
        x_focused = x[:, :, 9000:11000]

        global_output = self.global_view(x)
        focused_output = self.focused_view(x_focused)

        global_summary  = self.pool_branch(global_output)
        focused_summary = self.pool_branch(focused_output)
        
        merge_views = torch.cat([global_summary, focused_summary], dim = 1)

        aux_out = self.aux_branch(aux)

        return self.classifier(torch.cat([merge_views, aux_out], dim = 1))


class TESSModelV5(nn.Module):
    def __init__(self):
        super().__init__()

        # Wide-angle lens: Massive kernel to see the macro shape of the lightcurve
        # Final output length hitting the pool branch: 156 (156 / 4 = 39, perfectly divisible)
        self.global_view = nn.Sequential(
            nn.Conv1d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = 61,
                stride = 4,
                #padding = 30
            ),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Dropout1d(p = 0.2),
            nn.MaxPool1d(kernel_size = 4),
            nn.Conv1d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = 31,
                stride = 2,
                padding = 15
            ),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout1d(p = 0.2),
            nn.MaxPool1d(kernel_size = 4)
        )

        # Portrait lens: Medium kernel tailored to the 50-150 step width of a transit
        # Final output length hitting the pool branch: 248 (248 / 4 = 62, perfectly divisible)
        self.focused_view = nn.Sequential(
            nn.Conv1d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = 5,
                stride = 2,
                #padding = 10
            ),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Dropout1d(p = 0.2),
            nn.MaxPool1d(kernel_size = 2),
            nn.Conv1d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = 5,
                stride = 1,
                #padding = 3
            ), 
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout1d(p = 0.2),
            nn.MaxPool1d(kernel_size = 2)
        )

        # Classifier input size remains perfectly matched at 512 features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 128, out_features = 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(p = 0.5), 
            nn.Linear(in_features = 128, out_features = 1)
        )

    def pool_branch(self, feat):
        avg = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
        mn = -F.adaptive_max_pool1d(-feat, 1).squeeze(-1)
        return torch.cat([avg, mn], dim = 1)
        
    def forward(self, x):
        x_focused = x[:, :, 9000:11000]

        global_output = self.global_view(x)
        focused_output = self.focused_view(x_focused)

        global_summary  = self.pool_branch(global_output)
        focused_summary = self.pool_branch(focused_output)
        
        merge_views = torch.cat([global_summary, focused_summary], dim = 1)

        return self.classifier(merge_views)
