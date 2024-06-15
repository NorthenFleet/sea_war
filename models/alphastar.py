import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaStar(nn.Module):
    def __init__(self, input_shape, scalar_input_dim, entity_input_dim, n_actions, lstm_hidden_size=256):
        super(AlphaStar, self).__init__()

        # Spatial encoder using ResNet
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Scalar encoder
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # Entity encoder using Transformer
        self.entity_encoder = nn.Transformer(
            d_model=entity_input_dim, nhead=8, num_encoder_layers=3)

        # Core LSTM
        self.lstm = nn.LSTM(input_size=256 + 256 + 256,
                            hidden_size=lstm_hidden_size, batch_first=True)

        # Action type head with Residual MLP
        self.action_type_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        # Delay head
        self.delay_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Queued head
        self.queued_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Assuming binary classification
        )

        # Selected units head using Pointer Network
        self.selected_units_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, entity_input_dim)  # Number of entities
        )

        # Target unit head using Attention
        self.target_unit_head = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size, num_heads=8)

        # Target point head using Deconv ResNet
        self.target_point_head = nn.Sequential(
            nn.ConvTranspose2d(lstm_hidden_size, 128,
                               kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 2, kernel_size=3, stride=1,
                               padding=1)  # Assuming 2D coordinates
        )

        # Value network
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, spatial_input, scalar_input, entity_input):
        # Encode spatial input
        spatial_features = self.spatial_encoder(spatial_input)
        spatial_features = spatial_features.view(spatial_features.size(0), -1)

        # Encode scalar input
        scalar_features = self.scalar_encoder(scalar_input)

        # Encode entity input
        entity_features = self.entity_encoder(entity_input, entity_input)

        # Combine features
        combined_features = torch.cat(
            (spatial_features, scalar_features, entity_features), dim=-1)

        # LSTM core
        lstm_out, _ = self.lstm(combined_features.unsqueeze(0))
        lstm_out = lstm_out.squeeze(0)

        # Heads
        action_type_logits = self.action_type_head(lstm_out)
        delay_logits = self.delay_head(lstm_out)
        queued_logits = self.queued_head(lstm_out)
        selected_units_logits = self.selected_units_head(lstm_out)

        target_unit_attn_output, _ = self.target_unit_head(
            lstm_out.unsqueeze(0), lstm_out.unsqueeze(0), lstm_out.unsqueeze(0))
        target_point_logits = self.target_point_head(
            lstm_out.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        value = self.value_head(lstm_out)

        return {
            "action_type": action_type_logits,
            "delay": delay_logits,
            "queued": queued_logits,
            "selected_units": selected_units_logits,
            "target_unit": target_unit_attn_output.squeeze(0),
            "target_point": target_point_logits,
            "value": value
        }


# Example instantiation and forward pass
input_shape = (3, 64, 64)  # Example spatial input shape
scalar_input_dim = 10  # Example scalar input dimension
entity_input_dim = 20  # Example entity input dimension
n_actions = 10  # Example number of actions

model = AlphaStar(input_shape, scalar_input_dim, entity_input_dim, n_actions)
spatial_input = torch.randn(1, *input_shape)
scalar_input = torch.randn(1, scalar_input_dim)
# Example batch size of 1, sequence length of 10
entity_input = torch.randn(1, 10, entity_input_dim)

outputs = model(spatial_input, scalar_input, entity_input)

for key, value in outputs.items():
    print(f"{key}: {value.shape}")  # Print the shapes of the model outputs
