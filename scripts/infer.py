# Fix OpenMP library conflict - MUST be set before importing any ML libraries
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms

import pydicom
import openslide

# Reuse dataset and worker init from training
from .deprecated_train import WholeSlideDataset, _worker_init_fn


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _create_model(model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    name = model_name.lower()
    if name == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model.to(device)


def _load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt.get('model_name', 'resnet50')
    num_classes = ckpt.get('num_classes') or (ckpt['model_state_dict']['fc.weight'].shape[0] if 'fc.weight' in ckpt['model_state_dict'] else 2)
    class_names = ckpt.get('class_names') or [str(i) for i in range(num_classes)]
    model = _create_model(model_name, num_classes, device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, class_names


def _load_segmentation_model(checkpoint_path: Path, device: torch.device, num_classes_fallback: int = 1):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Build a DeepLabV3-ResNet50 head by default (matches training path)
    try:
        seg = models.segmentation.deeplabv3_resnet50(pretrained=False)
    except Exception:
        seg = models.segmentation.fcn_resnet50(pretrained=False)
    # Determine out channels
    out_ch = int(ckpt.get('num_classes', num_classes_fallback) or num_classes_fallback)
    # Replace classifier last conv to match
    try:
        in_ch = seg.classifier[-1].in_channels
        seg.classifier[-1] = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    except Exception:
        try:
            in_ch = seg.classifier[4].in_channels
            seg.classifier[4] = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        except Exception:
            pass
    seg.load_state_dict(ckpt['model_state_dict'], strict=False)
    seg.eval()
    return seg.to(device), out_ch


def _connected_components_bbox(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Simple 4-connected component analysis returning bounding boxes (x0,y0,x1,y1)."""
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    bboxes: List[Tuple[int, int, int, int]] = []
    # Offsets for 4-connectivity
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for y in range(h):
        row = mask[y]
        for x in range(w):
            if row[x] and not visited[y, x]:
                minx = maxx = x
                miny = maxy = y
                # BFS
                q = [(x, y)]
                visited[y, x] = True
                while q:
                    cx, cy = q.pop()
                    if cx < minx: minx = cx
                    if cy < miny: miny = cy
                    if cx > maxx: maxx = cx
                    if cy > maxy: maxy = cy
                    for dx, dy in neighbors:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and mask[ny, nx]:
                            visited[ny, nx] = True
                            q.append((nx, ny))
                bboxes.append((minx, miny, maxx + 1, maxy + 1))
    return bboxes


def _tile_coords(width: int, height: int, tile: int, stride: int) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for y in range(0, max(1, height - tile + 1), stride):
        for x in range(0, max(1, width - tile + 1), stride):
            coords.append((x, y))
    # ensure coverage at right/bottom edges
    if (width - tile) % stride != 0 and width > tile:
        for y in range(0, max(1, height - tile + 1), stride):
            coords.append((width - tile, y))
    if (height - tile) % stride != 0 and height > tile:
        for x in range(0, max(1, width - tile + 1), stride):
            coords.append((x, height - tile))
    if width > tile and height > tile:
        coords.append((width - tile, height - tile))
    # dedupe
    coords = list(dict.fromkeys(coords))
    return coords


@torch.no_grad()
def _segment_slide_svs(slide_path: str, model: nn.Module, device: torch.device, patch_size: int, stride: int, level: int,
                       transform: transforms.Compose, out_dir: Path,
                       cls_model: Optional[nn.Module] = None,
                       cls_transform: Optional[transforms.Compose] = None,
                       cls_names: Optional[List[str]] = None) -> Dict[str, Any]:
    slide = openslide.OpenSlide(slide_path)
    level_dims = slide.level_dimensions[level]
    W, H = level_dims
    coords = _tile_coords(W, H, patch_size, stride)
    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    total = len(coords)
    for i, (x, y) in enumerate(coords):
        img = slide.read_region((x, y), level, (patch_size, patch_size)).convert('RGB')
        inp = transform(img).unsqueeze(0).to(device)
        out = model(inp)
        logits = out['out'] if isinstance(out, dict) else out
        # binary or multi-class; use foreground probability
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        else:
            probs = torch.softmax(logits, dim=1)[0, 1].detach().cpu().numpy() if logits.shape[1] > 1 else torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        prob_map[y:y+patch_size, x:x+patch_size] += probs
        count_map[y:y+patch_size, x:x+patch_size] += 1.0
    # avoid divide by zero
    count_map[count_map == 0] = 1.0
    avg_prob = prob_map / count_map
    # threshold to mask
    mask = (avg_prob >= 0.5).astype(np.uint8)

    # Save mask and overlay
    base_name = os.path.splitext(os.path.basename(slide_path))[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_path = out_dir / f"{base_name}_mask_level{level}.png"
    mask_img.save(mask_path)

    thumb = slide.get_thumbnail((W, H)).convert('RGBA') if hasattr(slide, 'get_thumbnail') else Image.new('RGBA', (W, H))
    overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    red = Image.fromarray((mask * 200).astype(np.uint8))
    red_rgba = Image.merge('RGBA', (red, Image.new('L', red.size, 0), Image.new('L', red.size, 0), red))
    overlay = Image.alpha_composite(thumb, red_rgba)
    overlay_path = out_dir / f"{base_name}_overlay_level{level}.png"
    overlay.save(overlay_path)

    # ROIs as bounding boxes
    bboxes = _connected_components_bbox(mask.astype(bool))
    rois = []
    for (x0, y0, x1, y1) in bboxes:
        roi = {'bbox': [int(x0), int(y0), int(x1), int(y1)], 'area': int((x1 - x0) * (y1 - y0))}
        if cls_model is not None and (x1 > x0) and (y1 > y0):
            try:
                crop = slide.read_region((int(x0), int(y0)), level, (int(x1 - x0), int(y1 - y0))).convert('RGB')
                cimg = cls_transform(crop).unsqueeze(0).to(device) if cls_transform else transforms.ToTensor()(crop).unsqueeze(0).to(device)
                logits = cls_model(cimg)
                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
                pred_idx = int(np.argmax(probs))
                roi['cls_index'] = pred_idx
                roi['cls_name'] = (cls_names[pred_idx] if cls_names and 0 <= pred_idx < len(cls_names) else str(pred_idx))
                roi['cls_confidence'] = float(probs[pred_idx])
            except Exception:
                pass
    rois.append(roi)
    slide.close()
    return {
        'mask_path': str(mask_path),
        'overlay_path': str(overlay_path),
        'image_size': {'width': W, 'height': H},
        'rois': rois,
    }


@torch.no_grad()
def _segment_slide_dicom(dcm_path: str, model: nn.Module, device: torch.device, patch_size: int, stride: int,
                         transform: transforms.Compose, out_dir: Path,
                         cls_model: Optional[nn.Module] = None,
                         cls_transform: Optional[transforms.Compose] = None,
                         cls_names: Optional[List[str]] = None) -> Dict[str, Any]:
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    H, W = arr.shape[:2]
    coords = _tile_coords(W, H, patch_size, stride)
    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    for (x, y) in coords:
        patch = arr[y:y+patch_size, x:x+patch_size]
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            pad = np.zeros((patch_size, patch_size, 3), dtype=arr.dtype)
            pad[:patch.shape[0], :patch.shape[1]] = patch
            patch = pad
        img = Image.fromarray(patch.astype(np.uint8))
        inp = transform(img).unsqueeze(0).to(device)
        out = model(inp)
        logits = out['out'] if isinstance(out, dict) else out
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        else:
            probs = torch.softmax(logits, dim=1)[0, 1].detach().cpu().numpy() if logits.shape[1] > 1 else torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        prob_map[y:y+patch_size, x:x+patch_size] += probs
        count_map[y:y+patch_size, x:x+patch_size] += 1.0
    count_map[count_map == 0] = 1.0
    avg_prob = prob_map / count_map
    mask = (avg_prob >= 0.5).astype(np.uint8)

    base_name = os.path.splitext(os.path.basename(dcm_path))[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_path = out_dir / f"{base_name}_mask.png"
    mask_img.save(mask_path)

    base_rgb = Image.fromarray(arr.astype(np.uint8)).convert('RGBA')
    red = Image.fromarray((mask * 200).astype(np.uint8))
    red_rgba = Image.merge('RGBA', (red, Image.new('L', red.size, 0), Image.new('L', red.size, 0), red))
    overlay = Image.alpha_composite(base_rgb, red_rgba)
    overlay_path = out_dir / f"{base_name}_overlay.png"
    overlay.save(overlay_path)

    bboxes = _connected_components_bbox(mask.astype(bool))
    rois = []
    for (x0, y0, x1, y1) in bboxes:
        roi = {'bbox': [int(x0), int(y0), int(x1), int(y1)], 'area': int((x1 - x0) * (y1 - y0))}
        if cls_model is not None and (x1 > x0) and (y1 > y0):
            try:
                crop = arr[int(y0):int(y1), int(x0):int(x1)]
                if crop.shape[0] > 0 and crop.shape[1] > 0:
                    img = Image.fromarray(crop.astype(np.uint8)).convert('RGB')
                    cimg = cls_transform(img).unsqueeze(0).to(device) if cls_transform else transforms.ToTensor()(img).unsqueeze(0).to(device)
                    logits = cls_model(cimg)
                    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
                    pred_idx = int(np.argmax(probs))
                    roi['cls_index'] = pred_idx
                    roi['cls_name'] = (cls_names[pred_idx] if cls_names and 0 <= pred_idx < len(cls_names) else str(pred_idx))
                    roi['cls_confidence'] = float(probs[pred_idx])
            except Exception:
                pass
        rois.append(roi)
    return {
        'mask_path': str(mask_path),
        'overlay_path': str(overlay_path),
        'image_size': {'width': W, 'height': H},
        'rois': rois,
    }


def _discover_slides(input_path: str) -> List[str]:
    p = Path(input_path)
    if p.is_file():
        return [str(p)] if p.suffix.lower() in ['.svs', '.dcm'] else []
    slides: List[str] = []
    for root, dirs, files in os.walk(p):
        for f in files:
            if f.lower().endswith(('.svs', '.dcm')):
                slides.append(str(Path(root) / f))
    slides.sort()
    return slides


def _match_annotations(slide_paths: List[str], annotations_dir: Optional[str]) -> List[Optional[str]]:
    ann_dir = None
    if annotations_dir:
        ann_dir = annotations_dir if os.path.isabs(annotations_dir) else os.path.join(os.getcwd(), annotations_dir)
        if not os.path.exists(ann_dir):
            ann_dir = None
    else:
        cand1 = os.path.join(os.getcwd(), 'Annotations')
        ann_dir = cand1 if os.path.exists(cand1) else None

    ann_paths: List[Optional[str]] = []
    for sp in slide_paths:
        base = os.path.splitext(os.path.basename(sp))[0]
        found = None
        if ann_dir:
            cand = os.path.join(ann_dir, f"{base}.xml")
            if os.path.exists(cand):
                found = cand
        if not found:
            cand2 = os.path.splitext(sp)[0] + '.xml'
            if os.path.exists(cand2):
                found = cand2
        ann_paths.append(found)
    return ann_paths


@torch.no_grad()
def infer(
    input_path: str,
    checkpoint: str = 'checkpoints/best_model.pth',
    output_json: Optional[str] = None,
    batch_size: int = 16,
    patch_size: int = 224,
    patches_per_slide: int = 200,
    svs_level: int = 1,
    num_workers: int = 0,
    annotations_dir: Optional[str] = None,
    device: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[object] = None,
    task: str = 'classification',
    stride: Optional[int] = None,
    cls_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    device_t = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    slides = _discover_slides(input_path)
    if not slides:
        raise RuntimeError(f"No slides found in {input_path}")

    ann_paths = _match_annotations(slides, annotations_dir)
    num_annotated = sum(1 for a in ann_paths if a)

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        # fallback to latest if best not present
        alt = ckpt_path.parent / 'latest_checkpoint.pth' if ckpt_path.suffix else Path('checkpoints/latest_checkpoint.pth')
        ckpt_path = alt

    if task == 'segmentation':
        seg_model, out_ch = _load_segmentation_model(ckpt_path, device_t)
        class_names = ['background', 'lesion'] if out_ch == 2 else ['lesion']
        cls_model = None
        cls_names = None
        cls_transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        if cls_checkpoint:
            try:
                cls_model, cls_names = _load_checkpoint_model(Path(cls_checkpoint), device_t)
            except Exception:
                cls_model = None
                cls_names = None
    else:
        model, class_names = _load_checkpoint_model(ckpt_path, device_t)

    transform = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    results: List[Dict[str, Any]] = []
    if task == 'segmentation':
        out_mask_dir = Path('output') / 'segmentation'
        stride_eff = int(stride or max(32, patch_size // 2))
        for sp in slides:
            if sp.lower().endswith('.svs'):
                seg_info = _segment_slide_svs(sp, seg_model, device_t, patch_size, stride_eff, svs_level, transform, out_mask_dir,
                                              cls_model=cls_model, cls_transform=cls_transform, cls_names=cls_names)
            elif sp.lower().endswith('.dcm'):
                seg_info = _segment_slide_dicom(sp, seg_model, device_t, patch_size, stride_eff, transform, out_mask_dir,
                                                cls_model=cls_model, cls_transform=cls_transform, cls_names=cls_names)
            else:
                raise ValueError(f"Unsupported slide type: {sp}")

            slide_result: Dict[str, Any] = {
                'slide_path': sp,
                'overlay_path': seg_info['overlay_path'],
                'mask_path': seg_info['mask_path'],
                'rois': seg_info['rois'],
                'image_size': seg_info['image_size'],
                'annotation_used': bool(_match_annotations([sp], annotations_dir)[0]),
            }
            results.append(slide_result)

        summary = {
            'num_slides': len(slides),
            'num_annotated_slides': num_annotated,
            'task': 'segmentation',
            'checkpoint_used': str(ckpt_path),
            'device': str(device_t),
            'patch_size': patch_size,
            'stride': stride_eff,
            'svs_level': svs_level,
            'results': results,
        }
    else:
        labels = [0] * len(slides)  # dummy labels, not used
        dataset = WholeSlideDataset(
            slide_paths=slides,
            labels=labels,
            patch_size=patch_size,
            patches_per_slide=patches_per_slide,
            transform=transform,
            svs_level=svs_level,
            cache_patches=False,
            annotation_paths=ann_paths,
        )

        dl_kwargs = dict(batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
        if num_workers and num_workers > 0:
            dl_kwargs.update(dict(num_workers=num_workers, persistent_workers=True, prefetch_factor=2, worker_init_fn=_worker_init_fn))
        else:
            dl_kwargs.update(dict(num_workers=0))
        loader = DataLoader(dataset, **dl_kwargs)

        # Accumulators per slide index
        sums = torch.zeros((len(slides), len(class_names)), dtype=torch.float32, device=device_t)
        counts = torch.zeros((len(slides),), dtype=torch.long, device=device_t)

        patch_counter = 0
        total_patches = len(slides) * patches_per_slide
        for inputs, _ in loader:
            inputs = inputs.to(device_t, non_blocking=True)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            bsz = probs.size(0)
            for i in range(bsz):
                slide_idx = (patch_counter + i) // patches_per_slide
                if slide_idx >= len(slides):
                    continue
                sums[slide_idx] += probs[i]
                counts[slide_idx] += 1
            patch_counter += bsz
            # progress callback
            if progress_cb:
                try:
                    progress_cb(min(patch_counter, total_patches), total_patches)
                except Exception:
                    pass
            # cancel check
            if cancel_event is not None:
                try:
                    if getattr(cancel_event, 'is_set', lambda: False)():
                        break
                except Exception:
                    pass

        # Aggregate and prepare results
        for si, sp in enumerate(slides):
            if counts[si] == 0:
                avg = torch.full((len(class_names),), float('nan'))
                pred_idx = -1
                conf = float('nan')
            else:
                avg = (sums[si] / counts[si]).detach().cpu()
                pred_idx = int(torch.argmax(avg).item())
                conf = float(avg[pred_idx].item())
            results.append({
                'slide_path': sp,
                'pred_class': class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx),
                'pred_index': pred_idx,
                'confidence': conf,
                'probs': {cls: float(avg[j].item()) if counts[si] > 0 else None for j, cls in enumerate(class_names)},
                'patches_used': int(counts[si].item()),
                'annotation_used': bool(ann_paths[si]),
            })

        summary = {
            'num_slides': len(slides),
            'num_annotated_slides': num_annotated,
            'task': 'classification',
            'class_names': class_names,
            'checkpoint_used': str(ckpt_path),
            'device': str(device_t),
            'patch_size': patch_size,
            'patches_per_slide': patches_per_slide,
            'results': results,
        }

    # Save JSON
    out_path = None
    if output_json:
        out_path = Path(output_json)
    else:
        out_dir = Path('output')
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'inference_results.json'
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"Inference results written to {out_path}")
    except Exception as e:
        print(f"Could not write results JSON: {e}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Whole Slide Inference')
    parser.add_argument('--input', required=True, help='Path to a slide file or a directory to search for slides')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--output', default=None, help='Path to write JSON results (default: output/inference_results.json)')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--patch-size', type=int, default=224)
    parser.add_argument('--patches-per-slide', type=int, default=200)
    parser.add_argument('--svs-level', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--annotations-dir', default=None, help='Directory containing XML annotations (optional)')
    parser.add_argument('--device', default=None, help="'cuda' or 'cpu' (default: auto)")
    parser.add_argument('--task', choices=['classification', 'segmentation'], default='classification', help='Inference task')
    parser.add_argument('--stride', type=int, default=None, help='Segmentation stride (defaults to patch_size//2)')
    args = parser.parse_args()

    _ = infer(
        input_path=args.input,
        checkpoint=args.checkpoint,
        output_json=args.output,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        patches_per_slide=args.patches_per_slide,
        svs_level=args.svs_level,
        num_workers=args.num_workers,
        annotations_dir=args.annotations_dir,
        device=args.device,
        task=args.task,
        stride=args.stride,
    )


if __name__ == '__main__':
    # Windows/DataLoader safety
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except Exception:
        pass
    main()
