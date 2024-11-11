import matplotlib.pyplot as plt
import seaborn as sns

from item2vec.formers import PositionalEncoding

embed_dim = 128
max_len = 50
pos_encoding = PositionalEncoding(embed_dim=embed_dim, max_len=max_len, scale=100)
positional_encoding_data = pos_encoding.pe[0, :].cpu().numpy()

plt.figure(figsize=(12, 6))
sns.heatmap(positional_encoding_data, cmap="viridis", cbar=True)
plt.title("Positional Encoding Heatmap")
plt.xlabel("Embedding Dimension")
plt.ylabel("Position")
plt.show()
