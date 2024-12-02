import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops


class LightGCNConv(MessagePassing):
    def __init__(self):
        super(LightGCNConv, self).__init__(aggr="sum")

    def forward(self, x, edge_index):
        row, col = edge_index

        # Column degree normalization
        col_deg = degree(col, x.size(0))
        col_deg_inv_sqrt = col_deg.pow(-1)  # Use square root for normalization
        col_deg_inv_sqrt[col_deg_inv_sqrt == float("inf")] = 0.0
        col_norm = col_deg_inv_sqrt[col]

        # # Row degree normalization with log transformation
        # row_deg = degree(row, x.size(0))
        # log_row_deg = torch.log(1 + row_deg)  # Apply log to reduce skewness
        # row_deg_normalized = (log_row_deg / log_row_deg.sum()) * 10  # Normalize
        # row_norm = row_deg_normalized[row]

        # Combine normalizations
        norm = col_norm  # * row_norm
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # Calculate message
        msg = norm.view(-1, 1) * x_j
        return msg


# 예제 데이터
edge_index = torch.tensor(
    [
        [0, 1, 2, 1, 1, 1],
        [3, 3, 3, 3, 3, 3],
    ]
)
x = torch.tensor(
    [[1, 1, 0], [0.0, 0.3, 0], [0.5, 0, 0], [1, -1, 0]]
)  # 4개의 노드 임베딩

# LightGCNConv 실행
conv = LightGCNConv()

merge = []
for _ in range(1):
    xs = conv(x, edge_index)
    merge.append(xs)
    x = xs + x

print(merge)


# print("Updated Embeddings:")
# print(updated_x)
