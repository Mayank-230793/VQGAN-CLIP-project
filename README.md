## **Overview**  
This project is designed to develop modular AI building blocks leveraging generative AI technologies. It integrates advanced API calls to create flexible and reusable components for applications like image generation.

VQGAN stands for Vector Quantized Generative Adversarial Network, while CLIP stands for Contrastive Image-Language Pretraining. Whenever we say VQGAN-CLIP1, we refer to the interaction between these two networks. They’re separate models that work in tandem. The way they work is that VQGAN generates the images, while CLIP judges how well an image matches our text prompt. This interaction guides our generator to produce more accurate images.

CLIP guides VQGAN towards an image that is the best match to a given text. CLIP is the “Perceptor” and VQGAN is the “Generator”. VQGAN like all GANs VQGAN takes in a noise vector, and outputs a (realistic) image. CLIP on the other hand takes in an image and text, and outputs the image features and text features respectively. The similarity between image and text can be represented by the cosine similarity of the learnt feature vectors.

By leveraging CLIPs capacities as a “steering wheel”, we can use CLIP to guide a search through VQGAN’s latent space to find images that match a text prompt very well according to CLIP.

## **Features**  
- 🧩 Modular AI components for generative tasks.    
- 🛠️ Customizable pipelines for image generation.  
- ⚡ Optimized for high performance using pre-trained models.  
  


Here's a breakdown of the code with explanations of the important components:

### **1. Importing Libraries and Setup**

....

### **2. Downloading Pretrained Models**

```python
torch.hub.download_url_to_file('https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1', 'vqgan_imagenet_f16_16384.yaml')
torch.hub.download_url_to_file('https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1', 'vqgan_imagenet_f16_16384.ckpt')
```

- Downloads the **VQGAN model configuration** and **checkpoint** files from a URL. These are needed to load the VQGAN model for generating images.

### **3. Utilities for Image Resampling and Processing**

Functions like `sinc`, `lanczos`, `ramp`, and `resample` are used for resizing and resampling images using various interpolation techniques.

- **Lanczos resampling** (`lanczos`): Helps in resizing images by smoothing the pixels to reduce aliasing artifacts.
- **Resampling (`resample`)**: Used to resize an input tensor (image) to a specific size while maintaining the quality.

### **4. Custom Autograd Functions: `ReplaceGrad` and `ClampWithGrad`**

```python
class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward
    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)
```

- **ReplaceGrad**: Used to override the gradient flow during backpropagation, which is useful for certain transformations during optimization (e.g., preventing the backpropagation of some gradients).
  
```python
class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)
    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None
```

- **ClampWithGrad**: Custom gradient function that clamps the input tensor within a specified range during both the forward and backward passes.

### **5. Vector Quantization**

```python
def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)
```

- **Vector Quantization**: This process maps continuous values (in a latent space) to a discrete set of values (using a codebook). It is essential for models like VQGAN, which use a codebook of embeddings for representing images in a compact form.

### **6. Prompt Class for Text Prompts**

```python
class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))
```

- **Prompt**: This class is used to encode text prompts into embeddings using CLIP and then compute a loss for each prompt. The **weight** controls how much the prompt influences the generation, and **stop** defines when to stop optimizing a particular prompt.

### **7. MakeCutouts for Augmenting Images**

```python
class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'),
            K.RandomPerspective(0.7,p=0.7),
            K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
            K.RandomErasing((.1, .4), (.3, 1/.3), same_on_batch=True, p=0.7),
        )
```

- **MakeCutouts**: This is used to cut the image into multiple patches (or "cutouts") and apply augmentations like affine transforms, random perspective, and color jitter. This helps create a variety of representations of the image for training.

### **8. Image Resizing Utility**

```python
def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)
```

- **Resize Image**: Resizes an image while maintaining its aspect ratio.

### **9. Model Loading**

```python
def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
```

- **Load VQGAN Model**: Loads the VQGAN model based on the configuration file and checkpoint. This model is responsible for generating images.

### **10. Inference Function**

```python
def inference(text, seed, step_size, max_iterations, width, height, init_image, init_weight, target_images, cutn, cut_pow, video_file):
    all_frames = []
    size=[width, height]
    ...
    # Generate the image by optimizing z with respect to the loss functions
    return image
```

- **Inference Function**: This function orchestrates the process of generating an image from text prompts (or target images). It initializes a latent vector `z` (which is optimized over several iterations), applies cutouts, prompts, and optimizes the latent space representation.

- **Optimization**: The `z` latent vector is updated through backpropagation to minimize the losses related to the text and image prompts.


### **11. Running the Inference**

```python
img = inference(
    text = 'ghost in night with moon', 
    seed = 2,
    step_size = 0.12,
    max_iterations = 700,
    width = 512,
    height = 512,
    init_image = '',
    init_weight = 0.004,
    target_images = '', 
    cutn = 64,
    cut_pow = 0.3,
    video_file = "test1"
)
display_result(img)
```

- **Inference Run**: The image generation process starts with the text prompt `'ghost in night with moon'`. 

 image is done by iterating through various loss functions related to the text and image prompts.

---
