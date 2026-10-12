# NETT architecture specification

## Shared settings

- Six independently trained models: CNN, ViT, CNN2F, ViT2F, 3DCNN, ViViT.
- RGB images: height 160 × width 256. Single-frame input: `[B,3,160,256]`. Two-frame input: `[B,2,3,160,256]`, ordered previous, current.
- Convert `uint8` pixels to floating point and divide by 255. Floating-point inputs are already in `[0,1]`.
- All encoders use width 96, **three residual blocks**, and output `[B,128]`.
- All convolutions and linear layers use `bias=True`. Convolutions use `groups=1`, `dilation=1`, and zero padding. All LayerNorms use `eps=1e-6` and learned scale/bias.
- No pooling, CLS token, dropout, or batch normalization.

## Patch projection and positions

Apply the same patch-projection weights separately to each frame:

```python
nn.Conv2d(3, 96, kernel_size=16, stride=16, padding=0)
```

Each frame becomes `[B,96,10,16]`. Use internal layout `[B,T,10,16,96]`, where `T` is the number of frames processed together.

- Every model adds learned spatial positions of shape `[1,1,10,16,96]`.
- 3DCNN and ViViT also add learned temporal positions of shape `[1,2,1,1,96]`.
- CNN2F and ViT2F use the same spatial positions for both frames, with no temporal positions.

## Residual block

Each of the three blocks has separate weights and the following definition:

```python
norm1 = nn.LayerNorm(96, eps=1e-6)
norm2 = nn.LayerNorm(96, eps=1e-6)
ffn = nn.Sequential(
    nn.Linear(96, 384),
    nn.GELU(approximate="none"),
    nn.Linear(384, 96),
)

u = x + mixer(norm1(x))
y = u + ffn(norm2(u))
```

LayerNorm and the FFN operate on the last dimension independently at each position. The mixer preserves the input shape. There is no activation after either residual addition.

### Convolution mixers

| Models | Mixer | Layout used for convolution |
| --- | --- | --- |
| CNN, CNN2F | `nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1)` | `[B*T,96,10,16]` |
| 3DCNN | `nn.Conv3d(96,96,kernel_size=(3,3,3),stride=1,padding=1)` | `[B,96,2,10,16]` |

Restore the internal layout after convolution. 3DCNN retains two temporal positions through every block; temporal padding is zero.

### Attention mixer

ViT, ViT2F, and ViViT use three heads of width 32:

```python
qkv_proj = nn.Linear(96, 288)
out_proj = nn.Linear(96, 96)

# tokens: [M,N,96]
qkv = qkv_proj(tokens).reshape(M, N, 3, 3, 32)
q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
a = ((q @ k.transpose(-2, -1)) / math.sqrt(32)).softmax(dim=-1)
y = (a @ v).transpose(1, 2).reshape(M, N, 96)
y = out_proj(y)
```

No attention mask. Token order is time, height, width, with width varying fastest. Restore the internal layout after attention.

| Model | `M` | `N` | Attention scope |
| --- | --- | --- | --- |
| ViT | `B` | 160 | One frame |
| ViT2F | `2B` | 160 | Each frame independently |
| ViViT | `B` | 320 | Both frames jointly |

## Shared spatial readout

After block 3, apply `nn.LayerNorm(96, eps=1e-6)`. Restore channels-first maps and apply the following readout separately at each frame/time position, sharing weights between positions:

| Layer | Output shape |
| --- | --- |
| Reshape/permutation to channels first | `[B*T,96,10,16]` |
| `nn.Conv2d(96,32,kernel_size=(4,4),stride=(2,2),padding=0)` → `nn.ReLU()` | `[B*T,32,4,7]` |
| `nn.Conv2d(32,32,kernel_size=(4,3),stride=1,padding=0)` → `nn.ReLU()` | `[B*T,32,1,5]` |
| `flatten(start_dim=1)` | `[B*T,160]` |
| `nn.Linear(160,D)` | `[B*T,D]` |

`D=128` for single-frame models and `D=64` for two-frame models. The final map has one vertical and five horizontal positions. Flatten channels first, then horizontal positions within each channel.

For two-frame models, restore `[B,2,64]` and concatenate previous-position features before current-position features. Every model finishes with `nn.LayerNorm(128, eps=1e-6)`, producing `[B,128]`.

## Six model definitions

All models follow: patch projection → position embeddings → three residual blocks → final channel LayerNorm → spatial readout → concatenation if needed → output LayerNorm.

| Model | Frames | Backbone processing | Readout | Encoder parameters |
| --- | --- | --- | --- | ---: |
| CNN | Current | Three Conv2D-mixer blocks | 128 features | 644,640 |
| ViT | Current | Three attention-mixer blocks; 160 tokens | 128 features | 507,264 |
| CNN2F | Previous + current | Fold frames into batch; one shared CNN applied independently | Shared 64-feature readout per frame; concatenate | 634,336 |
| ViT2F | Previous + current | Fold frames into batch; one shared ViT applied independently; 160 tokens per frame | Shared 64-feature readout per frame; concatenate | 496,960 |
| 3DCNN | Previous + current | Three Conv3D-mixer blocks on `[B,2,10,16,96]` | Shared 64-feature readout per time position; concatenate | 1,132,192 |
| ViViT | Previous + current | Three attention-mixer blocks; 320 joint tokens | Shared 64-feature readout per time position; concatenate | 497,152 |

Counts include positions, normalization, and readout; they exclude actor/critic heads. CNN2F and ViT2F are frame-independent until the final `LayerNorm(128)`, whose statistics couple the concatenated halves. 3DCNN and ViViT mix frames inside every backbone block.

## Actor and critic MLPs

Share one visual encoder between actor and critic. Use separate MLP weights:

```python
actor = nn.Sequential(
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, A),
)
critic = nn.Sequential(
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 1),
)
```

`A` is the number of discrete actions. Return raw actor logits `[B,A]` and critic values `[B,1]`.

## Initialization

- Convolution/linear weights and biases: independently sample `Uniform(-1/sqrt(fan_in), +1/sqrt(fan_in))`.
- LayerNorm: scale 1, bias 0.
- Spatial/temporal position embeddings: `Normal(mean=0, std=0.02)`.
- Weights are shared across frames within a model, never across the six models.
