#%%
from test import *
# %% 
# this is a deterministic test for TransformerDecoderLayer
d_model = 4
nhead = 2
dim_feedforward = 16
dropout = 0.0
bsz = 2
seq_length = 5
tgt_length = 3

for batch_first in (True, False):
    def perm_fn(x):
        return x.transpose(1,0) if batch_first else x

    model = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, layer_norm_eps=0., batch_first=batch_first,)

    # set constant weights of the model
    for idx, p in enumerate(model.parameters()):
        x = p.data
        sz = x.view(-1).size(0)
        shape = x.shape
        x = torch.cos(torch.arange(0, sz).float().view(shape))
        p.data.copy_(x)

    # deterministic input
    decoder_input = torch.Tensor([[[20, 30, 40, 50]]])
    memory_input = torch.Tensor([[[60, 70, 80, 90]]])
    result = model(decoder_input, memory_input)
    ref_output = torch.Tensor([[[2.314351, 0.094805, -0.671322, 0.101977]]])
    result = result.detach().numpy()
    ref_output = ref_output.detach().numpy()
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    np.testing.assert_allclose(result, ref_output, atol=1e-5)

    # deterministic input
    decoder_input = perm_fn(torch.Tensor([[[9, 10, 11, 12]],
                                          [[11, 12, 13, 14]]]))
    memory_input = torch.Tensor([[[1, 2, 3, 4]]])
    result = model(decoder_input, memory_input)
    result = result.detach().numpy()
    ref_output = perm_fn(torch.Tensor([[[2.422245, 0.051716, -0.606338, -0.024756]],
                                        [[2.422245, 0.051716, -0.606338, -0.024756]]]))
    ref_output = ref_output.detach().numpy()
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    np.testing.assert_allclose(result, ref_output, atol=1e-5)

    # deterministic input
    decoder_input = perm_fn(torch.Tensor([[[1, 2, 3, 4]],
                                            [[5, 6, 7, 8]]]))
    memory_input = perm_fn(torch.Tensor([[[9, 10, 11, 12]],
                                            [[11, 12, 13, 14]]]))
    result = model(decoder_input, memory_input)
    ref_output = perm_fn(torch.Tensor([[[2.343536, 0.085561, -0.654954, 0.074991]],
                                        [[2.343536, 0.085561, -0.654954, 0.074991]]]))
    result = result.detach().numpy()
    ref_output = ref_output.detach().numpy()
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    np.testing.assert_allclose(result, ref_output, atol=1e-5)

    # deterministic input
    decoder_input = perm_fn(torch.Tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                        [0.2678, 0.3677, 0.4459, 0.7166]],
                                        [[0.8100, 0.3716, 0.4096, 0.1976],
                                        [0.6958, 0.8844, 0.6081, 0.8315]],
                                        [[0.0494, 0.9343, 0.5955, 0.3830],
                                        [0.5404, 0.3464, 0.9378, 0.6200]]]))
    memory_input = perm_fn(torch.Tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                        [0.5387, 0.1655, 0.3565, 0.0471]],
                                        [[0.8335, 0.2799, 0.5031, 0.2947],
                                        [0.1402, 0.0318, 0.7636, 0.1346]],
                                        [[0.6333, 0.9344, 0.1376, 0.9938],
                                        [0.8924, 0.2872, 0.6692, 0.2944]],
                                        [[0.9897, 0.6915, 0.3154, 0.1733],
                                        [0.8645, 0.3513, 0.3064, 0.0767]],
                                        [[0.8117, 0.2366, 0.4838, 0.7881],
                                        [0.3718, 0.4945, 0.9511, 0.0864]]]))
    result = model(decoder_input, memory_input)
    ref_output = perm_fn(torch.Tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                        [2.431935, 0.028907, -0.599809, -0.072488]],
                                        [[2.428457, 0.027053, -0.602275, -0.073462],
                                        [2.431970, 0.029387, -0.599789, -0.071621]],
                                        [[2.431934, 0.028196, -0.599802, -0.073809],
                                        [2.432306, 0.028858, -0.599542, -0.072846]]]))
    result = result.detach().numpy()
    ref_output = ref_output.detach().numpy()
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    np.testing.assert_allclose(result, ref_output, atol=1e-5)

    # key_padding_mask
    key_padding_mask = torch.zeros(2, 3) == 1
    result = model(decoder_input, memory_input, tgt_key_padding_mask=key_padding_mask)
    ref_output = perm_fn(torch.Tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                        [2.431935, 0.028907, -0.599809, -0.072488]],
                                        [[2.428457, 0.027053, -0.602275, -0.073462],
                                        [2.431970, 0.029387, -0.599789, -0.071621]],
                                        [[2.431934, 0.028196, -0.599802, -0.073809],
                                        [2.432306, 0.028858, -0.599542, -0.072846]]]))
    result = result.detach().numpy()
    ref_output = ref_output.detach().numpy()
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    np.testing.assert_allclose(result, ref_output, atol=1e-5)

    # key_padding_mask
    key_padding_mask[0, 2] = 1
    key_padding_mask[1, 1] = 1
    key_padding_mask[1, 2] = 1
    result = model(decoder_input, memory_input, tgt_key_padding_mask=key_padding_mask)
    ref_output = perm_fn(torch.Tensor([[[2.430025, 0.027643, -0.601164, -0.073476],
                                        [2.4323, 0.029375, -0.599553, -0.071881]],
                                        [[2.428523, 0.026838, -0.602226, -0.07391],
                                        [2.432634, 0.029842, -0.599318, -0.071253]],
                                        [[2.432278, 0.028152, -0.599555, -0.074139],
                                        [2.432659, 0.029244, -0.599294, -0.072382]]]))
    result = result.detach().numpy()
    ref_output = ref_output.detach().numpy()
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    np.testing.assert_allclose(result, ref_output, atol=1e-5)

    # memory_key_padding_mask
    key_padding_mask = torch.zeros(2, 5) == 1
    result = model(decoder_input, memory_input, memory_key_padding_mask=key_padding_mask)
    ref_output = perm_fn(torch.Tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                        [2.431935, 0.028907, -0.599809, -0.072488]],
                                        [[2.428457, 0.027053, -0.602275, -0.073462],
                                        [2.431970, 0.029387, -0.599789, -0.071621]],
                                        [[2.431934, 0.028196, -0.599802, -0.073809],
                                        [2.432306, 0.028858, -0.599542, -0.072846]]]))
    result = result.detach().numpy()
    ref_output = ref_output.detach().numpy()
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    np.testing.assert_allclose(result, ref_output, atol=1e-5)

    # memory_key_padding_mask
    key_padding_mask[0, 4] = 1
    key_padding_mask[1, 3] = 1
    key_padding_mask[1, 4] = 1
    result = model(decoder_input, memory_input, memory_key_padding_mask=key_padding_mask)
    ref_output = perm_fn(torch.Tensor([[[2.429757, 0.027358, -0.601351, -0.073816],
                                        [2.432692, 0.028583, -0.599263, -0.073634]],
                                        [[2.428247, 0.02662, -0.602419, -0.074123],
                                        [2.432657, 0.029055, -0.599293, -0.072732]],
                                        [[2.431515, 0.027687, -0.600096, -0.074459],
                                        [2.433075, 0.028543, -0.598987, -0.073985]]]))
    result = result.detach().numpy()
    ref_output = ref_output.detach().numpy()
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    np.testing.assert_allclose(result, ref_output, atol=1e-5)