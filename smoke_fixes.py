import sys, torch
sys.path.insert(0, 'C:/Users/Haziq/Documents/SNAP-C1/SNAP-C1')
from cortex.modules.diff_attention import DiffAttentionLayer
from cortex.modules.neural_memory import NeuralMemory
from cortex.model import EidosV1
print('Imports OK')

m = EidosV1(vocab_size=11203, d_model=512, n_heads=8, n_kv_heads=4, n_layers=4, dropout=0.0, num_values=torch.rand(11203))
x = torch.randint(0, 11203, (2, 32))
m.train()
m.neural_memory.reset()
out = m(x)
print(f'Forward OK')
# Test NaN guard in memory
m.neural_memory.M.data = torch.full((512, 512), float('nan'))
out2 = m(x)  # should reset M and not explode
print(f'After NaN M: forward OK, M NaN count={torch.isnan(m.neural_memory.M).sum().item()}')
print('All checks passed')
