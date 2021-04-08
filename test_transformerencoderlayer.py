#%%
from test import *
# %% TransformerEncoder
def get_a_test_layer(use_cuda, activation):
    d_model = 4
    nhead = 2
    dim_feedforward = 16
    dropout = 0.0
    device = torch.device("cuda" if use_cuda else "cpu")

    layer = TransformerEncoderLayer(
        d_model,
        nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation).to(device)

    with torch.no_grad():
        # set constant weights of the model
        for idx, p in enumerate(layer.parameters()):
            x = p.data
            sz = x.view(-1).size(0)
            shape = x.shape
            x = torch.cos(torch.arange(0, sz).float().view(shape))
            p.data.copy_(x)

    return layer

# this is a deterministic test for TransformerEncoder
activation = "relu"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

for batch_first in (True, False):
    def perm_fn(x):
        return x.transpose(1,0) if batch_first else x

    encoder_layer = get_a_test_layer(use_cuda=use_cuda, activation=activation)

    model = TransformerEncoder(encoder_layer, 1, batch_first=batch_first).to(device)

    # deterministic input
    encoder_input = perm_fn(torch.Tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
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
    result = model(encoder_input)
    ref_output = perm_fn(torch.Tensor([[[2.428589, 0.020835, -0.602055, -0.085249],
                                [2.427987, 0.021213, -0.602496, -0.084103]],
                            [[2.424689, 0.019155, -0.604793, -0.085672],
                                [2.413863, 0.022211, -0.612486, -0.072490]],
                            [[2.433774, 0.021598, -0.598343, -0.087548],
                                [2.425104, 0.019748, -0.604515, -0.084839]],
                            [[2.436185, 0.022682, -0.596625, -0.087261],
                                [2.433556, 0.021891, -0.598509, -0.086832]],
                            [[2.416246, 0.017512, -0.610712, -0.082961],
                                [2.422901, 0.024187, -0.606178, -0.074929]]]
    )).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-5)

    # all 0
    mask = torch.zeros([2, 5]).to(device) == 1
    result = model(encoder_input, src_key_padding_mask=mask)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-5)
    mask[0, 1] = 1
    mask[1, 3] = 1
    mask[1, 4] = 1
    result = model(encoder_input, src_key_padding_mask=mask)
    ref_output = perm_fn(torch.Tensor([[[2.429026, 0.020793, -0.601741, -0.085642],
                                [2.428811, 0.021445, -0.601912, -0.084252]],
                            [[2.425009, 0.019155, -0.604566, -0.085899],
                                [2.415408, 0.02249, -0.611415, -0.073]],
                            [[2.434199, 0.021682, -0.598039, -0.087699],
                                [2.42598, 0.019941, -0.603896, -0.085091]],
                            [[2.436457, 0.022736, -0.59643, -0.08736],
                                [2.434021, 0.022093, -0.598179, -0.08679]],
                            [[2.416531, 0.017498, -0.610513, -0.083181],
                                [2.4242, 0.024653, -0.605266, -0.074959]]]
    )).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-5)

    # test case 2, multiple layers no norm
    model = TransformerEncoder(encoder_layer, 2, batch_first=batch_first).to(device)
    result = model(encoder_input, src_key_padding_mask=mask)
    ref_output = perm_fn(torch.Tensor(
        [[[2.419051, 0.017446, -0.608738, -0.085003],
        [2.419102, 0.017452, -0.608703, -0.085026]],
        [[2.419043, 0.017445, -0.608744, -0.084999],
        [2.419052, 0.017446, -0.608738, -0.085004]],
        [[2.419067, 0.017448, -0.608727, -0.085010],
        [2.419098, 0.017452, -0.608706, -0.085024]],
        [[2.419072, 0.017449, -0.608724, -0.085012],
        [2.419119, 0.017455, -0.608691, -0.085034]],
        [[2.419019, 0.017442, -0.608761, -0.084989],
        [2.419075, 0.017449, -0.608722, -0.085014]]]
    )).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-5)

    model = TransformerEncoder(encoder_layer, 6, batch_first=batch_first).to(device)
    result = model(encoder_input, src_key_padding_mask=mask)
    ref_output = perm_fn(torch.Tensor(
        [[[2.419101, 0.017453, -0.608703, -0.085025],
        [2.419101, 0.017453, -0.608704, -0.085025]],
        [[2.419101, 0.017453, -0.608703, -0.085025],
        [2.419101, 0.017453, -0.608704, -0.085025]],
        [[2.419101, 0.017453, -0.608703, -0.085025],
        [2.419101, 0.017453, -0.608704, -0.085025]],
        [[2.419101, 0.017453, -0.608703, -0.085025],
        [2.419101, 0.017453, -0.608704, -0.085025]],
        [[2.419101, 0.017453, -0.608703, -0.085025],
        [2.419101, 0.017453, -0.608704, -0.085025]]]
    )).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-5)

    # test case 3, multiple layers with norm
    # d_model = 4
    norm = nn.LayerNorm(4)
    model = TransformerEncoder(encoder_layer, 2, norm=norm, batch_first=batch_first).to(device)
    result = model(encoder_input, src_key_padding_mask=mask)
    ref_output = perm_fn(torch.Tensor(
        [[[1.695949, -0.357635, -0.893077, -0.445238],
        [1.695955, -0.357639, -0.893050, -0.445266]],
        [[1.695948, -0.357634, -0.893082, -0.445233],
        [1.695950, -0.357635, -0.893077, -0.445238]],
        [[1.695951, -0.357636, -0.893069, -0.445246],
        [1.695955, -0.357639, -0.893052, -0.445264]],
        [[1.695952, -0.357636, -0.893066, -0.445249],
        [1.695957, -0.357641, -0.893041, -0.445276]],
        [[1.695946, -0.357632, -0.893095, -0.445220],
        [1.695952, -0.357637, -0.893065, -0.445251]]]
    )).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-5)

    model = TransformerEncoder(encoder_layer, 6, norm=norm, batch_first=batch_first).to(device)
    result = model(encoder_input, src_key_padding_mask=mask)
    ref_output = perm_fn(torch.Tensor(
        [[[1.695955, -0.357639, -0.893051, -0.445265],
        [1.695955, -0.357639, -0.893051, -0.445265]],
        [[1.695955, -0.357639, -0.893051, -0.445265],
        [1.695955, -0.357639, -0.893051, -0.445265]],
        [[1.695955, -0.357639, -0.893051, -0.445265],
        [1.695955, -0.357639, -0.893051, -0.445265]],
        [[1.695955, -0.357639, -0.893051, -0.445265],
        [1.695955, -0.357639, -0.893051, -0.445265]],
        [[1.695955, -0.357639, -0.893051, -0.445265],
        [1.695955, -0.357639, -0.893051, -0.445265]]]
    )).to(device)
    assertEqual(tuple(result.shape), tuple(ref_output.shape))
    torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-5)
# %%
