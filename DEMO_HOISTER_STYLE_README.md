# SL-HOI HOIster-Style Gradio Demo

This demo is an add-on for the SL-HOI repository:

https://github.com/MPI-Lab/SL-HOI

Copy these files into the root of an existing SL-HOI checkout:

```text
demo_gradio_hoister_style.py
cam_reader.py
gradio_utils/
```

The launcher resolves model files relative to the SL-HOI repo root by default,
so no hard-coded machine paths are required when the weights are stored in the
standard layout:

```text
weights/SL-HOI-weights/pretrained/hico/pytorch_model.bin
weights/SL-HOI-weights/pretrained/hico_ov/pytorch_model.bin
weights/SL-HOI-weights/pretrained/swig/pytorch_model.bin
weights/SL-HOI-weights/params/hico/classifier_eval.pt
weights/SL-HOI-weights/params/hico/classifier_default.pt
weights/SL-HOI-weights/params/swig/classifier_swig_dict.pt
```

Run HICO:

```bash
python demo_gradio_hoister_style.py --dataset hico --variant hico
```

Run HICO open-vocabulary checkpoint:

```bash
python demo_gradio_hoister_style.py --dataset hico --variant hico_ov
```

Run SWIG:

```bash
python demo_gradio_hoister_style.py --dataset swig --variant swig
```

If the demo file is not placed in the SL-HOI repo root, pass the repo root once:

```bash
python /path/to/demo_gradio_hoister_style.py --slhoi-root /path/to/SL-HOI
```

If the weights are stored elsewhere, override only the checkpoint/classifier
paths you changed:

```bash
python demo_gradio_hoister_style.py \
  --dataset hico \
  --variant hico \
  --ckpt /path/to/pytorch_model.bin \
  --classifier-eval /path/to/classifier_eval.pt \
  --classifier-train /path/to/classifier_default.pt
```

The demo predicts complete dataset triplets directly. It does not expose
arbitrary verb-object combinations and it does not show masks, because SL-HOI
does not predict masks.
