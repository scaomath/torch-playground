#%%
from test import *
# %%
def get_a_test_layer(use_cuda, activation):
    d_model = 4
    nhead = 2
    dim_feedforward = 16
    dropout = 0.0
    # layer_norm_eps = 0
    # first test of
    # 1e-7, 0.00012575089931488037
    # 0.,   0.00012575089931488037
    # 1e-5, 0.0001255124807357788
    # 1e-6, 0.00012575089931488037
    # 1e-4, 0.00012433528900146484
    # 1e-3, 0.00011134892702102661
    # 1e-2, 0.000179290771484375
    device = torch.device("cuda" if use_cuda else "cpu")

    layer = TransformerDecoderLayer(
        d_model,
        nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        # layer_norm_eps=layer_norm_eps,
        ).to(device)

    with torch.no_grad():
        # set constant weights of the model
        for idx, p in enumerate(layer.parameters()):
            x = p.data
            sz = x.view(-1).size(0)
            shape = x.shape
            x = torch.cos(torch.arange(0, sz).float().view(shape))
            p.data.copy_(x)

    return layer

#%%
# this is a deterministic test for TransformerDecoder

for batch_first in (False, True):
    def perm_fn(x):
        return x.transpose(1,0) if batch_first else x

    activation = "relu"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    decoder_layer = get_a_test_layer(use_cuda=use_cuda, activation=activation)
    model = TransformerDecoder(decoder_layer, 1, batch_first=batch_first).to(device)

    # deterministic input
    # decoder_input = torch.Tensor([[[20, 30, 40, 50]]]).to(device)
    # memory_input = torch.Tensor([[[60, 70, 80, 90]]]).to(device)
    # result = model(decoder_input, memory_input)
    # ref_output = torch.Tensor(
    #     [[[2.314351, 0.094805, -0.671322, 0.101977]]]).to(device)
    # assertEqual(tuple(result.shape), tuple(ref_output.shape))
    # torch.testing.assert_allclose(result, ref_output, rtol=1e-5, atol=1e-5)

    # deterministic input
    decoder_input = perm_fn(torch.Tensor([[[9, 10, 11, 12]],
                                [[11, 12, 13, 14]]])).to(device)
    memory_input = torch.Tensor([[[1, 2, 3, 4]]]).to(device)
    result = model(decoder_input, memory_input)
    ref_output = perm_fn(torch.Tensor(
        [[[2.422245, 0.051716, -0.606338, -0.024756]],
        [[2.422245, 0.051716, -0.606338, -0.024756]]]
    )).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    print(result, '\n\n', ref_output)
    torch.testing.assert_allclose(result, ref_output, rtol=1e-5, atol=1e-5)

    # deterministic input
    decoder_input = perm_fn(torch.Tensor([[[1, 2, 3, 4]],
                                [[5, 6, 7, 8]]])).to(device)
    memory_input = perm_fn(torch.Tensor([[[9, 10, 11, 12]],
                                [[11, 12, 13, 14]]])).to(device)
    result = model(decoder_input, memory_input)
    ref_output = perm_fn(torch.Tensor(
        [[[2.343536, 0.085561, -0.654954, 0.074991]],
        [[2.343536, 0.085561, -0.654954, 0.074991]]]
    )).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output, rtol=1e-5, atol=1e-5)

    # deterministic input
    decoder_input = perm_fn(torch.Tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                        [0.2678, 0.3677, 0.4459, 0.7166]],
                                        [[0.8100, 0.3716, 0.4096, 0.1976],
                                        [0.6958, 0.8844, 0.6081, 0.8315]],
                                        [[0.0494, 0.9343, 0.5955, 0.3830],
                                        [0.5404, 0.3464, 0.9378, 0.6200]]]
    )).to(device)
    memory_input = perm_fn(torch.Tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                        [0.5387, 0.1655, 0.3565, 0.0471]],
                                        [[0.8335, 0.2799, 0.5031, 0.2947],
                                        [0.1402, 0.0318, 0.7636, 0.1346]],
                                        [[0.6333, 0.9344, 0.1376, 0.9938],
                                        [0.8924, 0.2872, 0.6692, 0.2944]],
                                        [[0.9897, 0.6915, 0.3154, 0.1733],
                                        [0.8645, 0.3513, 0.3064, 0.0767]],
                                        [[0.8117, 0.2366, 0.4838, 0.7881],
                                        [0.3718, 0.4945, 0.9511, 0.0864]]]
    )).to(device)
    result = model(decoder_input, memory_input)
    ref_output = perm_fn(torch.Tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                        [2.431935, 0.028907, -0.599809, -0.072488]],
                                        [[2.428457, 0.027053, -0.602275, -0.073462],
                                        [2.431970, 0.029387, -0.599789, -0.071621]],
                                        [[2.431934, 0.028196, -0.599802, -0.073809],
                                        [2.432306, 0.028858, -0.599542, -0.072846]]]
    )).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output, rtol=1e-5, atol=1e-5)

    # key_padding_mask
    key_padding_mask = torch.zeros(2, 3).to(device) == 1
    result = model(decoder_input,
                memory_input,
                tgt_key_padding_mask=key_padding_mask)
    ref_output = perm_fn(torch.Tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                        [2.431935, 0.028907, -0.599809, -0.072488]],
                                        [[2.428457, 0.027053, -0.602275, -0.073462],
                                        [2.431970, 0.029387, -0.599789, -0.071621]],
                                        [[2.431934, 0.028196, -0.599802, -0.073809],
                                        [2.432306, 0.028858, -0.599542, -0.072846]]]
    )).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output, rtol=1e-5, atol=1e-5)

    # key_padding_mask
    key_padding_mask[0, 2] = 1
    key_padding_mask[1, 1] = 1
    key_padding_mask[1, 2] = 1
    result = model(decoder_input,
                memory_input,
                tgt_key_padding_mask=key_padding_mask)
    ref_output = perm_fn(torch.Tensor([[[2.430025, 0.027643, -0.601164, -0.073476],
                                        [2.4323, 0.029375, -0.599553, -0.071881]],
                                        [[2.428523, 0.026838, -0.602226, -0.07391],
                                        [2.432634, 0.029842, -0.599318, -0.071253]],
                                        [[2.432278, 0.028152, -0.599555, -0.074139],
                                        [2.432659, 0.029244, -0.599294, -0.072382]]]
    )).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output, rtol=1e-5, atol=1e-5)

    # memory_key_padding_mask
    key_padding_mask = torch.zeros(2, 5).to(device) == 1
    result = model(decoder_input,
                memory_input,
                memory_key_padding_mask=key_padding_mask)
    ref_output = perm_fn(torch.Tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                        [2.431935, 0.028907, -0.599809, -0.072488]],
                                        [[2.428457, 0.027053, -0.602275, -0.073462],
                                        [2.431970, 0.029387, -0.599789, -0.071621]],
                                        [[2.431934, 0.028196, -0.599802, -0.073809],
                                        [2.432306, 0.028858, -0.599542, -0.072846]]]
    )).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output, rtol=1e-5, atol=1e-5)

    # memory_key_padding_mask
    key_padding_mask[0, 4] = 1
    key_padding_mask[1, 3] = 1
    key_padding_mask[1, 4] = 1
    result = model(decoder_input,
                memory_input,
                memory_key_padding_mask=key_padding_mask)
    ref_output = perm_fn(torch.Tensor([[[2.429757, 0.027358, -0.601351, -0.073816],
                                        [2.432692, 0.028583, -0.599263, -0.073634]],
                                        [[2.428247, 0.02662, -0.602419, -0.074123],
                                        [2.432657, 0.029055, -0.599293, -0.072732]],
                                        [[2.431515, 0.027687, -0.600096, -0.074459],
                                        [2.433075, 0.028543, -0.598987, -0.073985]]]
    )).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output, rtol=1e-5, atol=1e-5)

    # multiple layers no norm
    model = TransformerDecoder(decoder_layer, 2, batch_first=batch_first).to(device)

    # deterministic input
    decoder_input = torch.Tensor([[[20, 30, 40, 50]]]).to(device)
    memory_input = torch.Tensor([[[60, 70, 80, 90]]]).to(device)
    result = model(decoder_input, memory_input)
    ref_output = torch.Tensor(
        [[[2.31316, 0.0950293, -0.671995, 0.102802]]]).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output, rtol=1e-5, atol=1e-5)

    # multiple layers no norm
    model = TransformerDecoder(decoder_layer, 6, batch_first=batch_first).to(device)

    # deterministic input
    decoder_input = perm_fn(torch.Tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                        [0.2678, 0.3677, 0.4459, 0.7166]],
                                        [[0.8100, 0.3716, 0.4096, 0.1976],
                                        [0.6958, 0.8844, 0.6081, 0.8315]],
                                        [[0.0494, 0.9343, 0.5955, 0.3830],
                                        [0.5404, 0.3464, 0.9378, 0.6200]]]
    )).to(device)
    memory_input = perm_fn(torch.Tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                        [0.5387, 0.1655, 0.3565, 0.0471]],
                                        [[0.8335, 0.2799, 0.5031, 0.2947],
                                        [0.1402, 0.0318, 0.7636, 0.1346]],
                                        [[0.6333, 0.9344, 0.1376, 0.9938],
                                        [0.8924, 0.2872, 0.6692, 0.2944]],
                                        [[0.9897, 0.6915, 0.3154, 0.1733],
                                        [0.8645, 0.3513, 0.3064, 0.0767]],
                                        [[0.8117, 0.2366, 0.4838, 0.7881],
                                        [0.3718, 0.4945, 0.9511, 0.0864]]]
    )).to(device)
    result = model(decoder_input, memory_input)
    ref_output = perm_fn(torch.Tensor([[[2.42794, 0.026164, -0.60263, -0.0747591],
                                        [2.43113, 0.0279516, -0.600376, -0.0736896]],
                                        [[2.42794, 0.026164, -0.60263, -0.0747591],
                                        [2.43113, 0.0279516, -0.600376, -0.0736896]],
                                        [[2.42794, 0.026164, -0.60263, -0.0747591],
                                        [2.43113, 0.0279516, -0.600376, -0.0736896]]]
    )).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output, rtol=1e-5, atol=1e-5)

    # multiple layers with norm
    # d_model = 4
    norm = nn.LayerNorm(4)
    model = TransformerDecoder(decoder_layer, 2, norm=norm, batch_first=batch_first).to(device)

    # deterministic input
    decoder_input = torch.Tensor([[[20, 30, 40, 50]]]).to(device)
    memory_input = torch.Tensor([[[60, 70, 80, 90]]]).to(device)
    result = model(decoder_input, memory_input)
    ref_output = torch.Tensor(
        [[[1.66166, -0.326986, -1.01466, -0.320017]]]).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output, rtol=1e-5, atol=1e-5)

    # multiple layers with norm
    model = TransformerDecoder(decoder_layer, 6, norm=norm, batch_first=batch_first).to(device)

    # deterministic input
    decoder_input = perm_fn(torch.Tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                        [0.2678, 0.3677, 0.4459, 0.7166]],
                                        [[0.8100, 0.3716, 0.4096, 0.1976],
                                        [0.6958, 0.8844, 0.6081, 0.8315]],
                                        [[0.0494, 0.9343, 0.5955, 0.3830],
                                        [0.5404, 0.3464, 0.9378, 0.6200]]]
    )).to(device)
    memory_input = perm_fn(torch.Tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                        [0.5387, 0.1655, 0.3565, 0.0471]],
                                        [[0.8335, 0.2799, 0.5031, 0.2947],
                                        [0.1402, 0.0318, 0.7636, 0.1346]],
                                        [[0.6333, 0.9344, 0.1376, 0.9938],
                                        [0.8924, 0.2872, 0.6692, 0.2944]],
                                        [[0.9897, 0.6915, 0.3154, 0.1733],
                                        [0.8645, 0.3513, 0.3064, 0.0767]],
                                        [[0.8117, 0.2366, 0.4838, 0.7881],
                                        [0.3718, 0.4945, 0.9511, 0.0864]]]
    )).to(device)
    result = model(decoder_input, memory_input)
    ref_output = perm_fn(torch.Tensor(
        [[[1.69559, -0.357291, -0.894741, -0.443553],
        [1.69571, -0.357363, -0.894154, -0.444196]],
        [[1.69559, -0.357291, -0.894741, -0.443553],
        [1.69571, -0.357363, -0.894154, -0.444196]],
        [[1.69559, -0.357291, -0.894741, -0.443553],
        [1.69571, -0.357363, -0.894154, -0.444196]]]
    )).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output, rtol=1e-5, atol=1e-5)

    # gelu activation test cases
    activation = "gelu"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    decoder_layer = get_a_test_layer(use_cuda=use_cuda, activation=activation)

    model = TransformerDecoder(decoder_layer, 1, batch_first=batch_first).to(device)

    # deterministic input
    decoder_input = torch.Tensor([[[20, 30, 40, 50]]]).to(device)
    memory_input = torch.Tensor([[[60, 70, 80, 90]]]).to(device)
    result = model(decoder_input, memory_input)
    ref_output = torch.Tensor([[[2.306435, 0.095946, -0.675796, 0.10687]]]
                            ).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output)

    # deterministic input
    decoder_input = perm_fn(torch.Tensor([[[9, 10, 11, 12]],
                                [[11, 12, 13, 14]]])).to(device)
    memory_input = perm_fn(torch.Tensor([[[1, 2, 3, 4]]])).to(device)
    result = model(decoder_input, memory_input)
    ref_output = perm_fn(torch.Tensor(
        [[[2.415448, 0.054389, -0.610932, -0.0156613]],
        [[2.415448, 0.054389, -0.610932, -0.0156613]]])).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output)

    # deterministic input
    decoder_input = perm_fn(torch.Tensor([[[1, 2, 3, 4]],
                                [[5, 6, 7, 8]]])).to(device)
    memory_input = perm_fn(torch.Tensor([[[9, 10, 11, 12]],
                                [[11, 12, 13, 14]]])).to(device)
    result = model(decoder_input, memory_input)
    ref_output = perm_fn(torch.Tensor(
        [[[2.338531, 0.087709, -0.65776, 0.080646]],
        [[2.338531, 0.087709, -0.65776, 0.080646]]])).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output)

    # deterministic input
    decoder_input = perm_fn(torch.Tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                        [0.2678, 0.3677, 0.4459, 0.7166]],
                                        [[0.8100, 0.3716, 0.4096, 0.1976],
                                        [0.6958, 0.8844, 0.6081, 0.8315]],
                                        [[0.0494, 0.9343, 0.5955, 0.3830],
                                        [0.5404, 0.3464, 0.9378, 0.6200]]]
    )).to(device)
    memory_input = perm_fn(torch.Tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                [0.5387, 0.1655, 0.3565, 0.0471]],
                                [[0.8335, 0.2799, 0.5031, 0.2947],
                                [0.1402, 0.0318, 0.7636, 0.1346]],
                                [[0.6333, 0.9344, 0.1376, 0.9938],
                                [0.8924, 0.2872, 0.6692, 0.2944]],
                                [[0.9897, 0.6915, 0.3154, 0.1733],
                                [0.8645, 0.3513, 0.3064, 0.0767]],
                                [[0.8117, 0.2366, 0.4838, 0.7881],
                                [0.3718, 0.4945, 0.9511, 0.0864]]]
    )).to(device)
    result = model(decoder_input, memory_input)
    ref_output = perm_fn(torch.Tensor(
                    [[[2.42049104, 0.03443088, -0.60793706, -0.05436271],
                    [2.42210631, 0.03546578, -0.60679895, -0.05357488]],
                    [[2.41907674, 0.0336104, -0.60892977, -0.05490462],
                    [2.42216881, 0.03586554, -0.6067524, -0.05289126]],
                    [[2.42205716, 0.03488046, -0.60683681, -0.05460596],
                    [2.42240309, 0.0354595, -0.60659063, -0.05378816]]]
        )).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output)
# %%
