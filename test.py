import numpy as np
import torch
from torch import nn
from libs.activation import *
from libs.transformer import *
from libs.my_utils import *
import libs.functional as F
torch.set_printoptions(7)


def test_Transformer_cell():
    # this is just a smoke test; these modules are implemented through
    # autograd so no Jacobian test is needed
    d_model = 512
    nhead = 16
    num_encoder_layers = 4
    num_decoder_layers = 3
    dim_feedforward = 256
    dropout = 0.3
    bsz = 8
    seq_length = 35
    tgt_length = 15
    for batch_first, src_size, tgt_size in zip((True, False),
                                                [(bsz, seq_length, d_model),
                                                (seq_length, bsz, d_model)],
                                                [(bsz, tgt_length, d_model),
                                                (tgt_length, bsz, d_model)]):
        transformer = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                        dim_feedforward, dropout, batch_first=batch_first)
        src = torch.randn(src_size)
        src_mask = transformer.generate_square_subsequent_mask(seq_length).double()
        tgt = torch.randn(tgt_size)
        tgt_mask = transformer.generate_square_subsequent_mask(tgt_length).double()
        memory_mask = torch.randn(tgt_length, seq_length).double()
        src_key_padding_mask = torch.rand(bsz, seq_length) >= 0.5
        tgt_key_padding_mask = torch.rand(bsz, tgt_length) >= 0.5
        memory_key_padding_mask = torch.rand(bsz, seq_length) >= 0.5

        output = transformer(src, tgt,
                                src_mask=src_mask,
                                tgt_mask=tgt_mask,
                                memory_mask=memory_mask,
                                src_key_padding_mask=src_key_padding_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask)
        output.sum().backward()

def test_transformerencoderlayer():
    # this is a deterministic test for TransformerEncoderLayer
    d_model = 4
    nhead = 2
    dim_feedforward = 16
    dropout = 0.0
    bsz = 2

    for batch_first in (False, True):
        def perm_fn(x):
            return x.transpose(1, 0) if batch_first else x

        model = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                            batch_first=batch_first)

        # set constant weights of the model
        for idx, p in enumerate(model.parameters()):
            x = p.data
            sz = x.view(-1).size(0)
            shape = x.shape
            x = torch.cos(torch.arange(0, sz).float().view(shape))
            p.data.copy_(x)

        # deterministic input
        encoder_input = torch.tensor([[[20., 30., 40., 50.]]])
        result = model(encoder_input)
        ref_output = torch.tensor([[[2.258703, 0.127985, -0.697881, 0.170862]]])
        result = result.detach().numpy()
        ref_output = ref_output.detach().numpy()
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)
        # 0 values are NOT masked. This shouldn't mask anything.
        mask = torch.tensor([[0]]) == 1
        result = model(encoder_input, src_key_padding_mask=mask)
        result = result.detach().numpy()
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)
        # 1 values are masked. Since there is only 1 input embedding this
        # will result in nan.
        mask = torch.tensor([[1]]) == 1
        result = model(encoder_input, src_key_padding_mask=mask)
        result = result.detach().numpy()
        assertTrue(np.isnan(result).all())

        # deterministic input
        encoder_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]],
                                                [[5., 6., 7., 8.]]]))
        result = model(encoder_input)
        ref_output = perm_fn(torch.tensor([[[2.272644, 0.119035, -0.691669, 0.153486]],
                                            [[2.272644, 0.119035, -0.691669, 0.153486]]]))
        result = result.detach().numpy()
        ref_output = ref_output.detach().numpy()
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)
        # all 0 which is no masking
        mask = torch.tensor([[0, 0]]) == 1
        result = model(encoder_input, src_key_padding_mask=mask)
        result = result.detach().numpy()
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)
        mask = torch.tensor([[1, 0]]) == 1
        result = model(encoder_input, src_key_padding_mask=mask)
        ref_output = perm_fn(torch.tensor([[[2.301516, 0.092249, -0.679101, 0.103088]],
                                            [[2.301516, 0.092249, -0.679101, 0.103088]]]))
        result = result.detach().numpy()
        ref_output = ref_output.detach().numpy()
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)

        # deterministic input
        encoder_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                [0.5387, 0.1655, 0.3565, 0.0471]],
                                                [[0.8335, 0.2799, 0.5031, 0.2947],
                                                [0.1402, 0.0318, 0.7636, 0.1346]],
                                                [[0.6333, 0.9344, 0.1376, 0.9938],
                                                [0.8924, 0.2872, 0.6692, 0.2944]],
                                                [[0.9897, 0.6915, 0.3154, 0.1733],
                                                [0.8645, 0.3513, 0.3064, 0.0767]],
                                                [[0.8117, 0.2366, 0.4838, 0.7881],
                                                [0.3718, 0.4945, 0.9511, 0.0864]]]))
        result = model(encoder_input)
        ref_output = perm_fn(torch.tensor([[[2.428589, 0.020835, -0.602055, -0.085249],
                                            [2.427987, 0.021213, -0.602496, -0.084103]],
                                            [[2.424689, 0.019155, -0.604793, -0.085672],
                                            [2.413863, 0.022211, -0.612486, -0.072490]],
                                            [[2.433774, 0.021598, -0.598343, -0.087548],
                                            [2.425104, 0.019748, -0.604515, -0.084839]],
                                            [[2.436185, 0.022682, -0.596625, -0.087261],
                                            [2.433556, 0.021891, -0.598509, -0.086832]],
                                            [[2.416246, 0.017512, -0.610712, -0.082961],
                                            [2.422901, 0.024187, -0.606178, -0.074929]]]))
        result = result.detach().numpy()
        ref_output = ref_output.detach().numpy()
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)
        # all 0
        mask = torch.zeros([2, 5]) == 1
        result = model(encoder_input, src_key_padding_mask=mask)
        result = result.detach().numpy()
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)
        mask[0, 1] = 1
        mask[1, 3] = 1
        mask[1, 4] = 1
        result = model(encoder_input, src_key_padding_mask=mask)
        ref_output = perm_fn(torch.tensor([[[2.429026, 0.020793, -0.601741, -0.085642],
                                            [2.428811, 0.021445, -0.601912, -0.084252]],
                                            [[2.425009, 0.019155, -0.604566, -0.085899],
                                            [2.415408, 0.02249 , -0.611415, -0.073]],
                                            [[2.434199, 0.021682, -0.598039, -0.087699],
                                            [2.42598, 0.019941, -0.603896, -0.085091]],
                                            [[2.436457, 0.022736, -0.59643 , -0.08736],
                                            [2.434021, 0.022093, -0.598179, -0.08679]],
                                            [[2.416531, 0.017498, -0.610513, -0.083181],
                                            [2.4242, 0.024653, -0.605266, -0.074959]]]))
        result = result.detach().numpy()
        ref_output = ref_output.detach().numpy()
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)

def test_transformerencoderlayer_gelu():
    # this is a deterministic test for TransformerEncoderLayer with gelu activation
    d_model = 4
    nhead = 2
    dim_feedforward = 16
    dropout = 0.0
    bsz = 2
    activation = "gelu"

    for batch_first in (True, False):
        def perm_fn(x):
            return x.transpose(1, 0) if batch_first else x

        model = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                            batch_first=batch_first)

        # set constant weights of the model
        for idx, p in enumerate(model.parameters()):
            x = p.data
            sz = x.view(-1).size(0)
            shape = x.shape
            x = torch.cos(torch.arange(0, sz).float().view(shape))
            p.data.copy_(x)

        # deterministic input
        encoder_input = torch.tensor([[[20., 30., 40., 50.]]])
        result = model(encoder_input)
        ref_output = torch.tensor([[[2.249815, 0.131006, -0.702199, 0.177868]]])
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output)

        # deterministic input
        encoder_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]],
                                                [[5., 6., 7., 8.]]]))
        result = model(encoder_input)
        ref_output = perm_fn(torch.tensor([[[2.264103, 0.121417, -0.696012, 0.159724]],
                                            [[2.264103, 0.121417, -0.696012, 0.159724]]]))
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output)

        # deterministic input
        encoder_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                [0.5387, 0.1655, 0.3565, 0.0471]],
                                                [[0.8335, 0.2799, 0.5031, 0.2947],
                                                [0.1402, 0.0318, 0.7636, 0.1346]],
                                                [[0.6333, 0.9344, 0.1376, 0.9938],
                                                [0.8924, 0.2872, 0.6692, 0.2944]],
                                                [[0.9897, 0.6915, 0.3154, 0.1733],
                                                [0.8645, 0.3513, 0.3064, 0.0767]],
                                                [[0.8117, 0.2366, 0.4838, 0.7881],
                                                [0.3718, 0.4945, 0.9511, 0.0864]]]))
        result = model(encoder_input)
        ref_output = perm_fn(torch.tensor([[[2.42163188, 0.03227153, -0.60714219, -0.05908082],
                                            [2.42151276, 0.03302179, -0.60722523, -0.05762651]],
                                            [[2.41926761, 0.02974034, -0.60879519, -0.0621269],
                                            [2.41626395, 0.03539356, -0.61087842, -0.04978623]],
                                            [[2.42382808, 0.03218872, -0.6055963, -0.06073591],
                                            [2.41983477, 0.03085259, -0.60840145, -0.06046414]],
                                            [[2.42500749, 0.03328855, -0.60476388, -0.0595334],
                                            [2.4237977, 0.03290575, -0.60561789, -0.05940082]],
                                            [[2.41383916, 0.02686345, -0.61256377, -0.06380707],
                                            [2.42000277, 0.03800944, -0.60824798, -0.04754947]]]))
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output)

def test_transformerdecoderlayer():
    # this is a deterministic test for TransformerDecoderLayer
    d_model = 4
    nhead = 2
    dim_feedforward = 16
    dropout = 0.0
    bsz = 2
    seq_length = 5
    tgt_length = 3

    for batch_first in (False, True):
        def perm_fn(x):
            return x.transpose(1, 0) if batch_first else x

        model = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                            batch_first=batch_first)

        # set constant weights of the model
        for idx, p in enumerate(model.parameters()):
            x = p.data
            sz = x.view(-1).size(0)
            shape = x.shape
            x = torch.cos(torch.arange(0, sz).float().view(shape))
            p.data.copy_(x)

        # deterministic input
        decoder_input = torch.tensor([[[20., 30., 40., 50.]]])
        memory_input = torch.tensor([[[60., 70., 80., 90.]]])
        result = model(decoder_input, memory_input)
        ref_output = torch.tensor([[[2.314351, 0.094805, -0.671322, 0.101977]]])
        result = result.detach().numpy()
        ref_output = ref_output.detach().numpy()
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[9., 10., 11., 12.]],
                                                [[11., 12., 13., 14.]]]))
        memory_input = torch.tensor([[[1., 2., 3., 4.]]])
        result = model(decoder_input, memory_input)
        result = result.detach().numpy()
        ref_output = perm_fn(torch.tensor([[[2.422245, 0.051716, -0.606338, -0.024756]],
                                            [[2.422245, 0.051716, -0.606338, -0.024756]]]))
        ref_output = ref_output.detach().numpy()
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]],
                                                [[5., 6., 7., 8.]]]))
        memory_input = perm_fn(torch.tensor([[[9., 10., 11., 12.]],
                                                [[11., 12., 13., 14.]]]))
        result = model(decoder_input, memory_input)
        ref_output = perm_fn(torch.tensor([[[2.343536, 0.085561, -0.654954, 0.074991]],
                                            [[2.343536, 0.085561, -0.654954, 0.074991]]]))
        result = result.detach().numpy()
        ref_output = ref_output.detach().numpy()
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                [0.2678, 0.3677, 0.4459, 0.7166]],
                                                [[0.8100, 0.3716, 0.4096, 0.1976],
                                                [0.6958, 0.8844, 0.6081, 0.8315]],
                                                [[0.0494, 0.9343, 0.5955, 0.3830],
                                                [0.5404, 0.3464, 0.9378, 0.6200]]]))
        memory_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
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
        ref_output = perm_fn(torch.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
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
        ref_output = perm_fn(torch.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
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
        ref_output = perm_fn(torch.tensor([[[2.430025, 0.027643, -0.601164, -0.073476],
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
        ref_output = perm_fn(torch.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
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
        ref_output = perm_fn(torch.tensor([[[2.429757, 0.027358, -0.601351, -0.073816],
                                            [2.432692, 0.028583, -0.599263, -0.073634]],
                                            [[2.428247, 0.02662, -0.602419, -0.074123],
                                            [2.432657, 0.029055, -0.599293, -0.072732]],
                                            [[2.431515, 0.027687, -0.600096, -0.074459],
                                            [2.433075, 0.028543, -0.598987, -0.073985]]]))
        result = result.detach().numpy()
        ref_output = ref_output.detach().numpy()
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)

def test_transformerdecoderlayer_gelu():
    # this is a deterministic test for TransformerDecoderLayer with gelu activation
    d_model = 4
    nhead = 2
    dim_feedforward = 16
    dropout = 0.0
    bsz = 2
    seq_length = 5
    tgt_length = 3
    activation = "gelu"

    for batch_first in (True, False):
        def perm_fn(x):
            return x.transpose(1, 0) if batch_first else x

        model = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                        batch_first=batch_first)

        # set constant weights of the model
        for idx, p in enumerate(model.parameters()):
            x = p.data
            sz = x.view(-1).size(0)
            shape = x.shape
            x = torch.cos(torch.arange(0, sz).float().view(shape))
            p.data.copy_(x)

        # deterministic input
        decoder_input = torch.tensor([[[20., 30., 40., 50.]]])
        memory_input = torch.tensor([[[60., 70., 80., 90.]]])
        result = model(decoder_input, memory_input)
        ref_output = torch.tensor([[[2.306435, 0.095946, -0.675796, 0.10687]]])
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[9., 10., 11., 12.]],
                                                [[11., 12., 13., 14.]]]))
        memory_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]]]))
        result = model(decoder_input, memory_input)
        ref_output = perm_fn(torch.tensor([[[2.415448, 0.054389, -0.610932, -0.0156613]],
                                            [[2.415448, 0.054389, -0.610932, -0.0156613]]]))
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]],
                                                [[5., 6., 7., 8.]]]))
        memory_input = perm_fn(torch.tensor([[[9., 10., 11., 12.]],
                                                [[11., 12., 13., 14.]]]))
        result = model(decoder_input, memory_input)
        ref_output = perm_fn(torch.tensor([[[2.338531, 0.087709, -0.65776, 0.080646]],
                                            [[2.338531, 0.087709, -0.65776, 0.080646]]]))
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                [0.2678, 0.3677, 0.4459, 0.7166]],
                                                [[0.8100, 0.3716, 0.4096, 0.1976],
                                                [0.6958, 0.8844, 0.6081, 0.8315]],
                                                [[0.0494, 0.9343, 0.5955, 0.3830],
                                                [0.5404, 0.3464, 0.9378, 0.6200]]]))
        memory_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
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
        ref_output = perm_fn(torch.tensor([[[2.42049104, 0.03443088, -0.60793706, -0.05436271],
                                            [2.42210631, 0.03546578, -0.60679895, -0.05357488]],
                                            [[2.41907674, 0.0336104, -0.60892977, -0.05490462],
                                            [2.42216881, 0.03586554, -0.6067524, -0.05289126]],
                                            [[2.42205716, 0.03488046, -0.60683681, -0.05460596],
                                            [2.42240309, 0.0354595, -0.60659063, -0.05378816]]]))
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output)

def test_transformerencoder():
    def get_a_test_layer(use_cuda, activation, batch_first=False):
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
            activation=activation,
            batch_first=batch_first).to(device)

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
            return x.transpose(1, 0) if batch_first else x

        encoder_layer = get_a_test_layer(use_cuda=use_cuda, activation=activation,
                                            batch_first=batch_first)

        model = TransformerEncoder(encoder_layer, 1).to(device)

        # deterministic input
        encoder_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
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
        ref_output = perm_fn(torch.tensor([[[2.428589, 0.020835, -0.602055, -0.085249],
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
        ref_output = perm_fn(torch.tensor([[[2.429026, 0.020793, -0.601741, -0.085642],
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
        model = TransformerEncoder(encoder_layer, 2).to(device)
        result = model(encoder_input, src_key_padding_mask=mask)
        ref_output = perm_fn(torch.tensor([[[2.419051, 0.017446, -0.608738, -0.085003],
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

        model = TransformerEncoder(encoder_layer, 6).to(device)
        result = model(encoder_input, src_key_padding_mask=mask)
        ref_output = perm_fn(torch.tensor([[[2.419101, 0.017453, -0.608703, -0.085025],
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
        model = TransformerEncoder(encoder_layer, 2, norm=norm).to(device)
        result = model(encoder_input, src_key_padding_mask=mask)
        ref_output = perm_fn(torch.tensor([[[1.695949, -0.357635, -0.893077, -0.445238],
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

        model = TransformerEncoder(encoder_layer, 6, norm=norm).to(device)
        result = model(encoder_input, src_key_padding_mask=mask)
        ref_output = perm_fn(torch.tensor([[[1.695955, -0.357639, -0.893051, -0.445265],
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


def test_transformerdecoder():
    def get_a_test_layer(use_cuda, activation, batch_first=False):
        d_model = 4
        nhead = 2
        dim_feedforward = 16
        dropout = 0.0
        device = torch.device("cuda" if use_cuda else "cpu")

        layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first).to(device)

        with torch.no_grad():
            # set constant weights of the model
            for idx, p in enumerate(layer.parameters()):
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = torch.cos(torch.arange(0, sz).float().view(shape))
                p.data.copy_(x)

        return layer

    # this is a deterministic test for TransformerDecoder
    for batch_first in (False, True):
        def perm_fn(x):
            return x.transpose(1, 0) if batch_first else x
        activation = "relu"
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        decoder_layer = get_a_test_layer(use_cuda=use_cuda, activation=activation,
                                            batch_first=batch_first)

        model = TransformerDecoder(decoder_layer, 1).to(device)

        # deterministic input
        decoder_input = torch.tensor([[[20., 30., 40., 50.]]]).to(device)
        memory_input = torch.tensor([[[60., 70., 80., 90.]]]).to(device)
        result = model(decoder_input, memory_input)
        ref_output = torch.tensor(
            [[[2.314351, 0.094805, -0.671322, 0.101977]]]).to(device)
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-3)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[9., 10., 11., 12.]],
                                                [[11., 12., 13., 14.]]])).to(device)
        memory_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]]])).to(device)
        result = model(decoder_input, memory_input)
        ref_output = perm_fn(torch.tensor(
            [[[2.422245, 0.051716, -0.606338, -0.024756]],
            [[2.422245, 0.051716, -0.606338, -0.024756]]]
        )).to(device)
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-4)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]],
                                    [[5., 6., 7., 8.]]])).to(device)
        memory_input = perm_fn(torch.tensor([[[9., 10., 11., 12.]],
                                    [[11., 12., 13., 14.]]])).to(device)
        result = model(decoder_input, memory_input)
        ref_output = perm_fn(torch.tensor(
            [[[2.343536, 0.085561, -0.654954, 0.074991]],
            [[2.343536, 0.085561, -0.654954, 0.074991]]]
        )).to(device)
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-4)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                [0.2678, 0.3677, 0.4459, 0.7166]],
                                                [[0.8100, 0.3716, 0.4096, 0.1976],
                                                [0.6958, 0.8844, 0.6081, 0.8315]],
                                                [[0.0494, 0.9343, 0.5955, 0.3830],
                                                [0.5404, 0.3464, 0.9378, 0.6200]]]
                                                )).to(device)
        memory_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
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
        ref_output = perm_fn(torch.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                            [2.431935, 0.028907, -0.599809, -0.072488]],
                                            [[2.428457, 0.027053, -0.602275, -0.073462],
                                            [2.431970, 0.029387, -0.599789, -0.071621]],
                                            [[2.431934, 0.028196, -0.599802, -0.073809],
                                            [2.432306, 0.028858, -0.599542, -0.072846]]]
                                            )).to(device)
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-5)

        # key_padding_mask
        key_padding_mask = torch.zeros(2, 3).to(device) == 1
        result = model(decoder_input,
                    memory_input,
                    tgt_key_padding_mask=key_padding_mask)
        ref_output = perm_fn(torch.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                            [2.431935, 0.028907, -0.599809, -0.072488]],
                                            [[2.428457, 0.027053, -0.602275, -0.073462],
                                            [2.431970, 0.029387, -0.599789, -0.071621]],
                                            [[2.431934, 0.028196, -0.599802, -0.073809],
                                            [2.432306, 0.028858, -0.599542, -0.072846]]]
                                            )).to(device)
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-5)

        # key_padding_mask
        key_padding_mask[0, 2] = 1
        key_padding_mask[1, 1] = 1
        key_padding_mask[1, 2] = 1
        result = model(decoder_input,
                    memory_input,
                    tgt_key_padding_mask=key_padding_mask)
        ref_output = perm_fn(torch.tensor([[[2.430025, 0.027643, -0.601164, -0.073476],
                                            [2.4323, 0.029375, -0.599553, -0.071881]],
                                            [[2.428523, 0.026838, -0.602226, -0.07391],
                                            [2.432634, 0.029842, -0.599318, -0.071253]],
                                            [[2.432278, 0.028152, -0.599555, -0.074139],
                                            [2.432659, 0.029244, -0.599294, -0.072382]]]
                                            )).to(device)
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-5)

        # memory_key_padding_mask
        key_padding_mask = torch.zeros(2, 5).to(device) == 1
        result = model(decoder_input,
                    memory_input,
                    memory_key_padding_mask=key_padding_mask)
        ref_output = perm_fn(torch.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                            [2.431935, 0.028907, -0.599809, -0.072488]],
                                            [[2.428457, 0.027053, -0.602275, -0.073462],
                                            [2.431970, 0.029387, -0.599789, -0.071621]],
                                            [[2.431934, 0.028196, -0.599802, -0.073809],
                                            [2.432306, 0.028858, -0.599542, -0.072846]]]
                                            )).to(device)
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-5)

        # memory_key_padding_mask
        key_padding_mask[0, 4] = 1
        key_padding_mask[1, 3] = 1
        key_padding_mask[1, 4] = 1
        result = model(decoder_input,
                    memory_input,
                    memory_key_padding_mask=key_padding_mask)
        ref_output = perm_fn(torch.tensor([[[2.429757, 0.027358, -0.601351, -0.073816],
                                            [2.432692, 0.028583, -0.599263, -0.073634]],
                                            [[2.428247, 0.02662, -0.602419, -0.074123],
                                            [2.432657, 0.029055, -0.599293, -0.072732]],
                                            [[2.431515, 0.027687, -0.600096, -0.074459],
                                            [2.433075, 0.028543, -0.598987, -0.073985]]]
                                            )).to(device)
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-5)

        # multiple layers no norm
        model = TransformerDecoder(decoder_layer, 2).to(device)

        # deterministic input
        decoder_input = torch.tensor([[[20., 30., 40., 50.]]]).to(device)
        memory_input = torch.tensor([[[60., 70., 80., 90.]]]).to(device)
        result = model(decoder_input, memory_input)
        ref_output = torch.tensor(
            [[[2.31316, 0.0950293, -0.671995, 0.102802]]]).to(device)
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-3)

        # multiple layers no norm
        model = TransformerDecoder(decoder_layer, 6).to(device)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                [0.2678, 0.3677, 0.4459, 0.7166]],
                                                [[0.8100, 0.3716, 0.4096, 0.1976],
                                                [0.6958, 0.8844, 0.6081, 0.8315]],
                                                [[0.0494, 0.9343, 0.5955, 0.3830],
                                                [0.5404, 0.3464, 0.9378, 0.6200]]]
                                                )).to(device)
        memory_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
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
        ref_output = perm_fn(torch.tensor([[[2.42794, 0.026164, -0.60263, -0.0747591],
                                            [2.43113, 0.0279516, -0.600376, -0.0736896]],
                                            [[2.42794, 0.026164, -0.60263, -0.0747591],
                                            [2.43113, 0.0279516, -0.600376, -0.0736896]],
                                            [[2.42794, 0.026164, -0.60263, -0.0747591],
                                            [2.43113, 0.0279516, -0.600376, -0.0736896]]]
                                            )).to(device)
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-5)

        # multiple layers with norm
        # d_model = 4
        norm = nn.LayerNorm(4)
        model = TransformerDecoder(decoder_layer, 2, norm=norm).to(device)

        # deterministic input
        decoder_input = torch.tensor([[[20., 30., 40., 50.]]]).to(device)
        memory_input = torch.tensor([[[60., 70., 80., 90.]]]).to(device)
        result = model(decoder_input, memory_input)
        ref_output = torch.tensor(
            [[[1.66166, -0.326986, -1.01466, -0.320017]]]).to(device)
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-3)

        # multiple layers with norm
        model = TransformerDecoder(decoder_layer, 6, norm=norm).to(device)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                [0.2678, 0.3677, 0.4459, 0.7166]],
                                                [[0.8100, 0.3716, 0.4096, 0.1976],
                                                [0.6958, 0.8844, 0.6081, 0.8315]],
                                                [[0.0494, 0.9343, 0.5955, 0.3830],
                                                [0.5404, 0.3464, 0.9378, 0.6200]]]
                                                )).to(device)
        memory_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
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
        ref_output = perm_fn(torch.tensor([[[1.69559, -0.357291, -0.894741, -0.443553],
                                            [1.69571, -0.357363, -0.894154, -0.444196]],
                                            [[1.69559, -0.357291, -0.894741, -0.443553],
                                            [1.69571, -0.357363, -0.894154, -0.444196]],
                                            [[1.69559, -0.357291, -0.894741, -0.443553],
                                            [1.69571, -0.357363, -0.894154, -0.444196]]]
                                            )).to(device)
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-5)

        # gelu activation test cases
        activation = "gelu"
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        decoder_layer = get_a_test_layer(use_cuda=use_cuda, activation=activation,
                                            batch_first=batch_first)

        model = TransformerDecoder(decoder_layer, 1).to(device)

        # deterministic input
        decoder_input = torch.tensor([[[20., 30., 40., 50.]]]).to(device)
        memory_input = torch.tensor([[[60., 70., 80., 90.]]]).to(device)
        result = model(decoder_input, memory_input)
        ref_output = torch.tensor([[[2.306435, 0.095946, -0.675796, 0.10687]]]
                                ).to(device)
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-3)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[9., 10., 11., 12.]],
                                                [[11., 12., 13., 14.]]])).to(device)
        memory_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]]])).to(device)
        result = model(decoder_input, memory_input)
        ref_output = perm_fn(torch.tensor(
            [[[2.415448, 0.054389, -0.610932, -0.0156613]],
            [[2.415448, 0.054389, -0.610932, -0.0156613]]])).to(device)
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-4)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]],
                                                [[5., 6., 7., 8.]]])).to(device)
        memory_input = perm_fn(torch.tensor([[[9., 10., 11., 12.]],
                                                [[11., 12., 13., 14.]]])).to(device)
        result = model(decoder_input, memory_input)
        ref_output = perm_fn(torch.tensor(
            [[[2.338531, 0.087709, -0.65776, 0.080646]],
            [[2.338531, 0.087709, -0.65776, 0.080646]]])).to(device)
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-4)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                [0.2678, 0.3677, 0.4459, 0.7166]],
                                                [[0.8100, 0.3716, 0.4096, 0.1976],
                                                [0.6958, 0.8844, 0.6081, 0.8315]],
                                                [[0.0494, 0.9343, 0.5955, 0.3830],
                                                [0.5404, 0.3464, 0.9378, 0.6200]]]
                                                )).to(device)
        memory_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
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
        ref_output = perm_fn(torch.tensor([[[2.42049104, 0.03443088, -0.60793706, -0.05436271],
                                            [2.42210631, 0.03546578, -0.60679895, -0.05357488]],
                                            [[2.41907674, 0.0336104, -0.60892977, -0.05490462],
                                            [2.42216881, 0.03586554, -0.6067524, -0.05289126]],
                                            [[2.42205716, 0.03488046, -0.60683681, -0.05460596],
                                            [2.42240309, 0.0354595, -0.60659063, -0.05378816]]]
                                            )).to(device)
        assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output, rtol=1e-7, atol=1e-5)