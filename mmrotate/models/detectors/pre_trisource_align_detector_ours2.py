"""Base-to-novel routing transfer built on TriSourceDetectorV16."""

from numbers import Integral

import torch
from mmrotate.core.bbox.transforms import obb2xyxy
import torch.distributed as dist
import torch.nn.functional as F

from ..builder import ROTATED_DETECTORS
from .pre_trisource_align_detector_v16 import TriSourceDetectorV16


@ROTATED_DETECTORS.register_module()
class TriSourceDetectorV16Ours2(TriSourceDetectorV16):
    """Ours with base routing signatures and novel-class routing regularization.

    Base signatures are collected from SAR GT boxes during base training
    (or an optional post-base pass) and stored in the checkpoint. Few-shot training retrieves
    signatures of semantically or visually similar base classes and adds a
    KL loss on novel GT-box tokens. The inherited inference path is unchanged.
    """

    def __init__(self, routing_transfer_cfg=None, **kwargs):
        # The V16 parent enforces the local SAR-CLIP encoder for Ours2 too.
        super().__init__(**kwargs)
        self.routing_transfer_cfg = dict(routing_transfer_cfg or {})
        self.base_routing_classes = list(
            self.routing_transfer_cfg.get('base_classes', []))
        self.sar_routing_class_names = list(
            self.routing_transfer_cfg.get('sar_class_names', []))
        self.novel_routing_classes = {
            self._canonical_name(name)
            for name in self.routing_transfer_cfg.get('novel_classes', [])
        }
        self.collect_base_signatures = bool(
            self.routing_transfer_cfg.get('collect_base_signatures', False))
        self.enable_routing_transfer = bool(
            self.routing_transfer_cfg.get('enable_transfer', False))
        self.routing_prior_mode = self.routing_transfer_cfg.get(
            'prior_mode', 'semantic_topk')
        self.routing_transfer_weight = float(
            self.routing_transfer_cfg.get('loss_weight', 0.1))
        self.routing_prior_temperature = max(
            float(self.routing_transfer_cfg.get('temperature', 0.07)), 1e-6)
        self.routing_prior_topk = max(
            int(self.routing_transfer_cfg.get('topk', 3)), 1)
        self.routing_prior_seed = int(
            self.routing_transfer_cfg.get('random_seed', 0))

        if self.routing_prior_mode not in (
                'none', 'semantic_topk', 'visual_topk',
                'uniform_base', 'random_base'):
            raise ValueError(
                'routing_transfer_cfg.prior_mode must be one of '
                'none, semantic_topk, visual_topk, uniform_base, or '
                'random_base; got '
                f'{self.routing_prior_mode!r}.')

        self._routing_gates = [
            module for module in self.backbone.modules()
            if hasattr(module, 'last_routing_probs')
        ]
        if (self.collect_base_signatures or self.enable_routing_transfer) and not self._routing_gates:
            raise RuntimeError(
                'Ours2 requires a backbone router that exposes routing probabilities.')

        if self.collect_base_signatures or self.enable_routing_transfer:
            for gate in self._routing_gates:
                gate.capture_routing_probs = True

        if self.collect_base_signatures and not self.base_routing_classes:
            raise ValueError(
                'routing_transfer_cfg.base_classes is required when collecting '
                'base routing signatures.')
        if self.enable_routing_transfer and not self.sar_routing_class_names:
            raise ValueError(
                'routing_transfer_cfg.sar_class_names is required for novel '
                'routing transfer.')

        num_layers = len(self._routing_gates)
        num_base_classes = len(self.base_routing_classes)
        num_experts = int(self._routing_gates[0].n) if self._routing_gates else 0
        if any(int(gate.n) != num_experts for gate in self._routing_gates):
            raise ValueError('All Ours2 routing layers must have the same expert count.')
        self.register_buffer(
            'base_routing_sums',
            torch.zeros(num_layers, num_base_classes, num_experts,
                        dtype=torch.float32))
        self.register_buffer(
            'base_routing_counts',
            torch.zeros(num_layers, num_base_classes, dtype=torch.float32))
        backbone_channels = getattr(self.backbone, 'channels', None)
        out_indices = getattr(self.backbone, 'out_indices', None)
        visual_dim = 0
        if backbone_channels:
            visual_index = (out_indices[-1] if out_indices else
                            len(backbone_channels) - 1)
            if visual_index < 0:
                visual_index += len(backbone_channels)
            if 0 <= visual_index < len(backbone_channels):
                visual_dim = int(backbone_channels[visual_index])
        self.register_buffer(
            'base_visual_sums',
            torch.zeros(num_base_classes, visual_dim, dtype=torch.float32))
        self.register_buffer(
            'base_visual_counts',
            torch.zeros(num_base_classes, dtype=torch.float32))
        self.register_buffer(
            'novel_visual_prototypes',
            torch.zeros(len(self.sar_routing_class_names), visual_dim,
                        dtype=torch.float32))
        self.register_buffer(
            'novel_visual_sums',
            torch.zeros(len(self.sar_routing_class_names), visual_dim,
                        dtype=torch.float32))
        self.register_buffer(
            'novel_visual_counts',
            torch.zeros(len(self.sar_routing_class_names),
                        dtype=torch.float32))
        self.register_buffer(
            'novel_visual_valid',
            torch.zeros(len(self.sar_routing_class_names), dtype=torch.bool))
        novel_visual_path = self.routing_transfer_cfg.get(
            'novel_visual_prototypes_path')
        if novel_visual_path:
            self._load_novel_visual_prototypes(novel_visual_path)
        self._routing_transfer_loss = None
        self._routing_signature_collection_only = False
        self._novel_visual_prototype_collection_only = False
        self._current_visual_feature_map = None

        if self.routing_prior_mode == 'visual_topk' and visual_dim <= 0:
            raise RuntimeError(
                'visual_topk requires the backbone to expose its output '
                'channel dimensions.')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Keep config-loaded support prototypes when loading the init ckpt."""
        if self.routing_transfer_cfg.get('novel_visual_prototypes_path'):
            for name in (
                    'novel_visual_prototypes', 'novel_visual_sums',
                    'novel_visual_counts', 'novel_visual_valid'):
                state_dict.pop(prefix + name, None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    def _load_novel_visual_prototypes(self, path):
        payload = torch.load(path, map_location='cpu')
        source_names = payload.get('class_names', [])
        source_prototypes = payload.get('prototypes')
        if source_prototypes is None:
            raise ValueError(
                f'Visual prototype file {path} has no `prototypes` tensor.')
        source_prototypes = torch.as_tensor(source_prototypes).float()
        if (source_prototypes.dim() != 2 or
                source_prototypes.size(0) != len(source_names) or
                source_prototypes.size(1) != self.novel_visual_prototypes.size(1)):
            raise ValueError(
                f'Incompatible visual prototype shape in {path}: '
                f'{tuple(source_prototypes.shape)} for classes {source_names}.')
        name_to_id = {
            self._canonical_name(name): index
            for index, name in enumerate(self.sar_routing_class_names)
        }
        for source_index, class_name in enumerate(source_names):
            class_index = name_to_id.get(self._canonical_name(class_name))
            if class_index is None:
                continue
            self.novel_visual_prototypes[class_index].copy_(
                F.normalize(source_prototypes[source_index], dim=0))
            self.novel_visual_valid[class_index] = True

    def set_base_signature_collection(self, enabled, in_eval=False):
        """Enable signature collection, optionally while the model is eval."""
        self.collect_base_signatures = bool(enabled)
        capture = self.collect_base_signatures or self.enable_routing_transfer
        for gate in self._routing_gates:
            gate.capture_routing_probs = capture
            gate.capture_routing_probs_in_eval = bool(
                enabled and in_eval)

    @torch.no_grad()
    def reset_base_routing_signatures(self):
        self.base_routing_sums.zero_()
        self.base_routing_counts.zero_()
        self.base_visual_sums.zero_()
        self.base_visual_counts.zero_()

    @torch.no_grad()
    def reset_novel_visual_prototypes(self):
        self.novel_visual_prototypes.zero_()
        self.novel_visual_sums.zero_()
        self.novel_visual_counts.zero_()
        self.novel_visual_valid.zero_()

    def set_novel_visual_prototype_collection(self, enabled):
        self._novel_visual_prototype_collection_only = bool(enabled)

    @torch.no_grad()
    def get_novel_visual_prototypes(self):
        novel_class_ids = [
            class_id for class_id, name in enumerate(self.sar_routing_class_names)
            if self._canonical_name(name) in self.novel_routing_classes
        ]
        if not novel_class_ids:
            raise RuntimeError('No novel classes are configured for visual prototypes.')
        missing = [
            self.sar_routing_class_names[class_id]
            for class_id in novel_class_ids
            if self.novel_visual_counts[class_id] <= 0
        ]
        if missing:
            raise RuntimeError(
                'No SAR support-region visual features were collected for: ' +
                ', '.join(missing))
        prototypes = self.novel_visual_sums / self.novel_visual_counts.clamp_min(
            1.0).unsqueeze(-1)
        prototypes = F.normalize(prototypes, dim=-1)
        self.novel_visual_prototypes.copy_(prototypes)
        self.novel_visual_valid.zero_()
        self.novel_visual_valid[novel_class_ids] = True
        return {
            'class_names': [self.sar_routing_class_names[index]
                            for index in novel_class_ids],
            'prototypes': prototypes[novel_class_ids].detach().cpu(),
            'counts': self.novel_visual_counts[novel_class_ids].detach().cpu(),
        }

    @staticmethod
    def _canonical_name(name):
        return ''.join(char.lower() for char in str(name) if char.isalnum())

    @staticmethod
    def _to_hbb(boxes):
        if boxes.size(-1) == 4:
            return boxes[:, :4]
        if boxes.size(-1) == 5:
            return obb2xyxy(boxes, version='le90')
        raise ValueError(
            f'Ours2 expects 4D or 5D GT boxes, got shape {tuple(boxes.shape)}.')

    def _base_label_to_signature_row(self):
        """Map SAR label IDs to checkpoint signature rows, preserving config order."""
        names = {self._canonical_name(name): label_id
                 for label_id, name in enumerate(self.sar_routing_class_names)}
        rows = {}
        for row, base_class in enumerate(self.base_routing_classes):
            label_id = (int(base_class) if isinstance(base_class, Integral)
                        else names.get(self._canonical_name(base_class)))
            if label_id is None or not 0 <= label_id < len(self.sar_routing_class_names):
                raise ValueError(f'Invalid base routing class: {base_class!r}.')
            if label_id in rows:
                raise ValueError(f'Duplicate base routing label ID: {label_id}.')
            rows[label_id] = row
        return rows

    def _base_class_name(self, signature_row):
        base_class = self.base_routing_classes[signature_row]
        if isinstance(base_class, Integral):
            return self.sar_routing_class_names[int(base_class)]
        return base_class

    @staticmethod
    def _sar_batch_offset(batch_inputs, datasets):
        offset = 0
        for dataset_name, tensor in zip(datasets, batch_inputs):
            if dataset_name == 'sar':
                return offset
            offset += int(tensor.size(0))
        return None

    @staticmethod
    def _box_token_mask(route_map, box, image_height, image_width):
        _, height, width, _ = route_map.shape
        device = route_map.device
        box = box.to(device=device, dtype=torch.float32)
        grid_x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5)
        grid_x = grid_x * (float(image_width) / width)
        grid_y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5)
        grid_y = grid_y * (float(image_height) / height)
        if box.numel() == 5:
            center_x, center_y, box_width, box_height, angle = box
            dx = grid_x[None, :] - center_x
            dy = grid_y[:, None] - center_y
            # Inverse rotation into the GT box's local coordinate frame.
            local_x = dx * angle.cos() + dy * angle.sin()
            local_y = -dx * angle.sin() + dy * angle.cos()
            mask = ((local_x.abs() <= box_width * 0.5) &
                    (local_y.abs() <= box_height * 0.5))
        elif box.numel() == 4:
            x1, y1, x2, y2 = box
            left, right = torch.minimum(x1, x2), torch.maximum(x1, x2)
            top, bottom = torch.minimum(y1, y2), torch.maximum(y1, y2)
            center_x, center_y = (left + right) * 0.5, (top + bottom) * 0.5
            mask = ((grid_y[:, None] >= top) & (grid_y[:, None] <= bottom) &
                    (grid_x[None, :] >= left) & (grid_x[None, :] <= right))
        else:
            raise ValueError('Routing regions require a 4D HBB or 5D OBB.')
        if not mask.any():
            # Guarantee a contribution for objects smaller than one token.
            distance = ((grid_y[:, None] - center_y) ** 2 +
                        (grid_x[None, :] - center_x) ** 2)
            mask.view(-1)[distance.argmin()] = True
        return mask

    def _iter_sar_gt_regions(self, batch_inputs, datasets, gt_bboxes,
                             gt_labels, route_maps):
        if (not isinstance(gt_bboxes, dict) or not isinstance(gt_labels, dict) or
                'sar' not in gt_bboxes or 'sar' not in gt_labels):
            return
        offset = self._sar_batch_offset(batch_inputs, datasets)
        if offset is None:
            return
        sar_images = next((tensor for name, tensor in zip(datasets, batch_inputs)
                           if name == 'sar'), None)
        if sar_images is None:
            return
        image_height, image_width = sar_images.shape[-2:]
        sar_boxes = gt_bboxes['sar']
        sar_labels = gt_labels['sar']
        num_images = min(len(sar_boxes), len(sar_labels), sar_images.size(0))
        for image_index in range(num_images):
            boxes = sar_boxes[image_index]
            labels = sar_labels[image_index]
            if not torch.is_tensor(boxes) or not torch.is_tensor(labels):
                continue
            num_boxes = min(boxes.size(0), labels.numel())
            for box_index in range(num_boxes):
                sample_index = offset + image_index
                box = boxes[box_index]
                class_id = int(labels.reshape(-1)[box_index].item())
                yield sample_index, image_height, image_width, box, class_id

    @torch.no_grad()
    def _record_base_signatures(self, batch_inputs, datasets, gt_bboxes,
                                gt_labels):
        if self.base_routing_sums.numel() == 0:
            return
        layer_maps = [gate.last_routing_probs for gate in self._routing_gates]
        if any(route_map is None for route_map in layer_maps):
            return

        batch_sums = torch.zeros_like(self.base_routing_sums)
        batch_counts = torch.zeros_like(self.base_routing_counts)
        base_rows = self._base_label_to_signature_row()
        for sample_index, image_h, image_w, box, class_id in self._iter_sar_gt_regions(
                batch_inputs, datasets, gt_bboxes, gt_labels, layer_maps):
            signature_row = base_rows.get(class_id)
            if signature_row is None:
                continue
            for layer_index, route_map in enumerate(layer_maps):
                if sample_index >= route_map.size(0):
                    continue
                mask = self._box_token_mask(
                    route_map, box, image_h, image_w)
                instance_signature = route_map[sample_index][mask].mean(dim=0)
                batch_sums[layer_index, signature_row] += instance_signature
                batch_counts[layer_index, signature_row] += 1.0

        # Make every rank's checkpoint carry the same global signature sums.
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_counts, op=dist.ReduceOp.SUM)
        self.base_routing_sums.add_(batch_sums)
        self.base_routing_counts.add_(batch_counts)

    def _base_signatures(self):
        if self.base_routing_counts.numel() == 0:
            raise RuntimeError('Ours2 has no configured base routing classes.')
        valid = self.base_routing_counts > 0
        if not valid.any():
            raise RuntimeError(
                'No base routing signatures were loaded. Train the Ours2 base '
                'model with collect_base_signatures=True, or run '
                'tools/misc/collect_ours2_routing_signatures.py for an old checkpoint, '
                'and use the collected checkpoint for few-shot fine-tuning.')
        if (valid.sum(dim=1) == 0).any():
            raise RuntimeError(
                'At least one MoE layer has no collected base routing '
                'signatures; check the Ours2 base checkpoint and router setup.')
        if not valid.all():
            missing = [
                f'layer {layer_id}: {self.base_routing_classes[class_id]}'
                for layer_id, class_id in torch.nonzero(
                    ~valid, as_tuple=False).tolist()
            ]
            raise RuntimeError(
                'Missing base routing signatures for ' + ', '.join(missing) +
                '. Run the SAR-only routing signature collection pass on the '
                'trained base checkpoint.')
        signatures = self.base_routing_sums / self.base_routing_counts.clamp_min(1.0).unsqueeze(-1)
        signatures = signatures / signatures.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return signatures, valid

    def _base_visual_prototypes(self):
        if self.base_visual_sums.size(-1) == 0:
            raise RuntimeError(
                'Ours2 has no configured SAR visual feature dimension. '
                'Use a backbone exposing channels and out_indices.')
        valid = self.base_visual_counts > 0
        if not bool(valid.all()):
            missing = [self.base_routing_classes[index]
                       for index in torch.nonzero(
                           ~valid, as_tuple=False).flatten().tolist()]
            raise RuntimeError(
                'Missing SAR visual prototypes for base classes: ' +
                ', '.join(missing) +
                '. Re-run SAR-only base signature collection.')
        prototypes = self.base_visual_sums / self.base_visual_counts.clamp_min(
            1.0).unsqueeze(-1)
        return F.normalize(prototypes, dim=-1)

    def _visual_base_weights(self, visual_query, valid_base_ids):
        prototypes = self._base_visual_prototypes()
        query = F.normalize(visual_query.detach().float(), dim=-1)
        similarities = prototypes[valid_base_ids] @ query
        topk = min(self.routing_prior_topk, similarities.numel())
        top_values, top_positions = similarities.topk(topk)
        selected_classes = valid_base_ids[top_positions]
        weights = F.softmax(top_values / self.routing_prior_temperature, dim=0)
        return selected_classes, weights

    def _backbone_visual_feature_map(self, result):
        """Return the final pre-neck backbone map in NCHW format."""
        features = result
        if (isinstance(result, tuple) and len(result) == 2 and
                isinstance(result[1], dict)):
            features = result[0]
        if isinstance(features, (tuple, list)):
            if not features:
                return None
            features = features[-1]
        if not torch.is_tensor(features) or features.dim() != 4:
            return None
        expected_dim = self.base_visual_sums.size(-1)
        if features.size(1) == expected_dim:
            return features
        if features.size(-1) == expected_dim:
            return features.permute(0, 3, 1, 2).contiguous()
        raise RuntimeError(
            'Could not identify the channel dimension in the final backbone '
            f'feature map {tuple(features.shape)} (expected {expected_dim}).')

    @torch.no_grad()
    def _record_base_visual_prototypes(self, batch_inputs, datasets,
                                       gt_bboxes, gt_labels,
                                       visual_feature_map):
        if (self.base_visual_sums.numel() == 0 or
                visual_feature_map is None):
            return
        visual_map = visual_feature_map.detach().permute(0, 2, 3, 1)
        batch_sums = torch.zeros_like(self.base_visual_sums)
        batch_counts = torch.zeros_like(self.base_visual_counts)
        base_rows = self._base_label_to_signature_row()
        for sample_index, image_h, image_w, box, class_id in self._iter_sar_gt_regions(
                batch_inputs, datasets, gt_bboxes, gt_labels, []):
            signature_row = base_rows.get(class_id)
            if signature_row is None:
                continue
            if sample_index >= visual_map.size(0):
                continue
            mask = self._box_token_mask(visual_map, box, image_h, image_w)
            descriptor = visual_map[sample_index][mask].float().mean(dim=0)
            if not torch.isfinite(descriptor).all():
                continue
            batch_sums[signature_row] += F.normalize(descriptor, dim=0)
            batch_counts[signature_row] += 1.0
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_counts, op=dist.ReduceOp.SUM)
        self.base_visual_sums.add_(batch_sums)
        self.base_visual_counts.add_(batch_counts)

    @torch.no_grad()
    def _record_novel_visual_prototypes(self, batch_inputs, datasets,
                                        gt_bboxes, gt_labels,
                                        visual_feature_map):
        if self.novel_visual_sums.numel() == 0 or visual_feature_map is None:
            return
        visual_map = visual_feature_map.detach().permute(0, 2, 3, 1)
        batch_sums = torch.zeros_like(self.novel_visual_sums)
        batch_counts = torch.zeros_like(self.novel_visual_counts)
        for sample_index, image_h, image_w, box, class_id in self._iter_sar_gt_regions(
                batch_inputs, datasets, gt_bboxes, gt_labels, []):
            if not 0 <= class_id < len(self.sar_routing_class_names):
                continue
            class_name = self.sar_routing_class_names[class_id]
            if self._canonical_name(class_name) not in self.novel_routing_classes:
                continue
            if sample_index >= visual_map.size(0):
                continue
            mask = self._box_token_mask(visual_map, box, image_h, image_w)
            descriptor = visual_map[sample_index][mask].float().mean(dim=0)
            if not torch.isfinite(descriptor).all():
                continue
            batch_sums[class_id] += F.normalize(descriptor, dim=0)
            batch_counts[class_id] += 1.0
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_counts, op=dist.ReduceOp.SUM)
        self.novel_visual_sums.add_(batch_sums)
        self.novel_visual_counts.add_(batch_counts)

    def _semantic_base_weights(self, novel_name, valid_base_ids, device):
        if self.prototype_aligner is None or not hasattr(
                self.prototype_aligner, 'text_feats_sar'):
            return None
        text_features = self.prototype_aligner.text_feats_sar
        name_to_id = {
            self._canonical_name(name): index
            for index, name in enumerate(self.sar_routing_class_names)
        }
        novel_id = name_to_id.get(self._canonical_name(novel_name))
        if novel_id is None or novel_id >= text_features.size(0):
            return None
        base_ids = []
        for base_id in valid_base_ids.tolist():
            if base_id >= len(self.base_routing_classes):
                continue
            text_id = name_to_id.get(
                self._canonical_name(self._base_class_name(base_id)))
            if text_id is not None and text_id < text_features.size(0):
                base_ids.append((base_id, text_id))
        if not base_ids:
            return None
        text_features = F.normalize(text_features.to(device).float(), dim=-1)
        base_class_ids = torch.tensor(
            [item[0] for item in base_ids], device=device, dtype=torch.long)
        base_text_ids = torch.tensor(
            [item[1] for item in base_ids], device=device, dtype=torch.long)
        similarities = text_features[base_text_ids] @ text_features[novel_id]
        topk = min(self.routing_prior_topk, similarities.numel())
        top_values, top_positions = similarities.topk(topk)
        selected_classes = base_class_ids[top_positions]
        weights = F.softmax(top_values / self.routing_prior_temperature, dim=0)
        return selected_classes, weights

    def _prior_for_novel(self, novel_name, novel_index, layer_index,
                         signatures, valid, visual_query=None):
        valid_base_ids = torch.nonzero(valid[layer_index], as_tuple=False).flatten()
        device = signatures.device
        if valid_base_ids.numel() == 0:
            return None

        if self.routing_prior_mode == 'uniform_base':
            selected_classes = valid_base_ids
            weights = signatures.new_full(
                (valid_base_ids.numel(),), 1.0 / valid_base_ids.numel())
        elif self.routing_prior_mode == 'random_base':
            generator = torch.Generator(device='cpu')
            generator.manual_seed(self.routing_prior_seed + novel_index)
            positions = torch.randperm(
                valid_base_ids.numel(), generator=generator)[:min(
                    self.routing_prior_topk, valid_base_ids.numel())]
            selected_classes = valid_base_ids[positions.to(valid_base_ids.device)]
            weights = signatures.new_full(
                (selected_classes.numel(),), 1.0 / selected_classes.numel())
        elif self.routing_prior_mode == 'visual_topk':
            if visual_query is None:
                raise RuntimeError(
                    'visual_topk did not receive a SAR GT-region feature.')
            selected_classes, weights = self._visual_base_weights(
                visual_query, valid_base_ids)
        else:
            semantic = self._semantic_base_weights(
                novel_name, valid_base_ids, device)
            if semantic is None:
                raise RuntimeError(
                    'Could not match CLIP text embeddings for novel class '
                    f'{novel_name!r} and the configured base classes. Check '
                    'routing_transfer_cfg.sar_class_names against align_cfg.sar_category.')
            selected_classes, weights = semantic

        prior = (signatures[layer_index, selected_classes] *
                 weights.to(device=device, dtype=signatures.dtype).unsqueeze(-1)).sum(0)
        return prior / prior.sum().clamp_min(1e-12)

    def _compute_routing_transfer_loss(self, batch_inputs, datasets,
                                       gt_bboxes, gt_labels):
        if not self.enable_routing_transfer or self.routing_transfer_weight <= 0:
            return None
        layer_maps = [gate.last_routing_probs for gate in self._routing_gates]
        if any(route_map is None for route_map in layer_maps):
            raise RuntimeError(
                'Ours2 could not capture router probabilities during training. '
                'Check that the configured backbone uses FusionOriginGate.')

        signatures, valid = self._base_signatures()
        class_name_to_index = {
            self._canonical_name(name): index
            for index, name in enumerate(self.sar_routing_class_names)
        }
        novel_class_indices = {
            class_name_to_index[self._canonical_name(name)]: (name, index)
            for index, name in enumerate(self.sar_routing_class_names)
            if self._canonical_name(name) in self.novel_routing_classes
        }
        if not novel_class_indices:
            novel_class_indices = {
                index: (name, index)
                for index, name in enumerate(self.sar_routing_class_names)
                if self._canonical_name(name) not in {
                    self._canonical_name(self._base_class_name(row))
                    for row in range(len(self.base_routing_classes))
                }
            }
        if not novel_class_indices:
            return None

        layer_losses = []
        per_layer_token_losses = [[] for _ in layer_maps]
        prior_cache = {}
        visual_feature_map = self._current_visual_feature_map
        if (self.routing_prior_mode == 'visual_topk' and
                visual_feature_map is None):
            raise RuntimeError(
                'visual_topk could not access the current SAR backbone feature map.')
        visual_map = (visual_feature_map.detach().permute(0, 2, 3, 1)
                      if visual_feature_map is not None else None)
        for sample_index, image_h, image_w, box, class_id in self._iter_sar_gt_regions(
                batch_inputs, datasets, gt_bboxes, gt_labels, layer_maps):
            novel_info = novel_class_indices.get(class_id)
            if novel_info is None:
                continue
            novel_name, novel_index = novel_info
            visual_query = None
            if self.routing_prior_mode == 'visual_topk':
                has_support_prototype = (
                    novel_index < self.novel_visual_valid.numel() and
                    bool(self.novel_visual_valid[novel_index].item()))
                if has_support_prototype:
                    visual_query = self.novel_visual_prototypes[novel_index]
                else:
                    if sample_index >= visual_map.size(0):
                        continue
                    visual_mask = self._box_token_mask(
                        visual_map, box, image_h, image_w)
                    visual_query = visual_map[sample_index][visual_mask].float().mean(dim=0)
                    if not torch.isfinite(visual_query).all():
                        continue
            for layer_index, route_map in enumerate(layer_maps):
                if sample_index >= route_map.size(0):
                    continue
                prior_key = (class_id, layer_index)
                cache_visual_prior = (
                    self.routing_prior_mode == 'visual_topk' and
                    novel_index < self.novel_visual_valid.numel() and
                    bool(self.novel_visual_valid[novel_index].item()))
                if (not cache_visual_prior or
                        prior_key not in prior_cache):
                    prior_cache[prior_key] = self._prior_for_novel(
                        novel_name, novel_index, layer_index,
                        signatures, valid, visual_query=visual_query)
                prior = prior_cache[prior_key]
                if prior is None:
                    continue
                mask = self._box_token_mask(route_map, box, image_h, image_w)
                token_probs = route_map[sample_index][mask].float().clamp_min(1e-12)
                token_probs = token_probs / token_probs.sum(
                    dim=-1, keepdim=True).clamp_min(1e-12)
                prior = prior.float().clamp_min(1e-12)
                prior = prior / prior.sum().clamp_min(1e-12)
                kl = (prior * (prior.log().unsqueeze(0) - token_probs.log())).sum(-1)
                per_layer_token_losses[layer_index].append(kl)

        for token_losses in per_layer_token_losses:
            if token_losses:
                layer_losses.append(torch.cat(token_losses).mean())
        if not layer_losses:
            # Keep the loss key present on every DDP rank, including ranks
            # whose local few-shot batch has no novel GT instances.
            return sum(route_map.sum() * 0.0 for route_map in layer_maps) / len(layer_maps)
        return torch.stack(layer_losses).mean()

    def _forward_backbone(self, batch_inputs, datasets, roi_features=None,
                          text_features=None, gt_bboxes=None, gt_labels=None):
        result = super()._forward_backbone(
            batch_inputs,
            datasets,
            roi_features=roi_features,
            text_features=text_features,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels)
        self._routing_transfer_loss = None
        visual_feature_map = None
        if (self.collect_base_signatures or
                self._novel_visual_prototype_collection_only or
                (self.training and self.enable_routing_transfer and
                 self.routing_prior_mode == 'visual_topk')):
            visual_feature_map = self._backbone_visual_feature_map(result)
        if self.collect_base_signatures:
            self._record_base_signatures(
                batch_inputs, datasets, gt_bboxes, gt_labels)
            self._record_base_visual_prototypes(
                batch_inputs, datasets, gt_bboxes, gt_labels,
                visual_feature_map)
        if self._novel_visual_prototype_collection_only:
            self._record_novel_visual_prototypes(
                batch_inputs, datasets, gt_bboxes, gt_labels,
                visual_feature_map)
        if self.training and self.enable_routing_transfer:
            self._current_visual_feature_map = visual_feature_map
            self._routing_transfer_loss = self._compute_routing_transfer_loss(
                batch_inputs, datasets, gt_bboxes, gt_labels)
            self._current_visual_feature_map = None
        if ((self.training and self.enable_routing_transfer) or
                self.collect_base_signatures):
            for gate in self._routing_gates:
                gate.last_routing_probs = None
        return result

    @torch.no_grad()
    def _collect_base_signature_batch(self, img, img_metas, gt_bboxes,
                                      gt_labels, gt_bboxes_ignore=None,
                                      **kwargs):
        """Collect signatures from a SAR-only batch without running heads."""
        if gt_bboxes_ignore is not None:
            raise ValueError(
                'Ours2 signature collection expects gt_bboxes_ignore=None.')

        img = self.gather_dict_values(img)
        self.gather_dict_values(img_metas)
        gt_bboxes = self.gather_dict_values(gt_bboxes, ignore_tensor=True)
        gt_labels = self.gather_dict_values(gt_labels, ignore_tensor=True)
        sar_images = img['sar']
        if not torch.is_tensor(sar_images) or sar_images.size(0) == 0:
            raise RuntimeError(
                'The signature collection loader returned a batch without SAR images.')
        if img.get('rgb', []) or img.get('ifr', []):
            raise RuntimeError(
                'The signature collection loader must contain SAR data only.')

        batch_inputs = [sar_images]
        datasets = ['sar']
        backbone_gt_bboxes = dict(
            sar=gt_bboxes['sar'], rgb=[], ifr=[])
        backbone_gt_labels = dict(
            sar=gt_labels['sar'], rgb=[], ifr=[])

        roi_features, text_features = (None, None)
        if self.enable_backbone_align_inputs:
            roi_features, text_features = self._collect_backbone_align_inputs(
                kwargs)
            if (self.auto_backbone_text_from_aligner and
                    text_features is None):
                text_features = self._build_backbone_text_features_from_aligner(
                    sar_images.device)
            roi_features = self._move_align_features_to_device(
                roi_features, sar_images.device)
            text_features = self._move_align_features_to_device(
                text_features, sar_images.device)

        self._forward_backbone(
            batch_inputs,
            datasets,
            roi_features=roi_features,
            text_features=text_features,
            gt_bboxes=backbone_gt_bboxes,
            gt_labels=backbone_gt_labels)
        return {'loss_signature_collection': sar_images.new_zeros(())}

    @torch.no_grad()
    def _collect_novel_visual_prototype_batch(
            self, img, img_metas, gt_bboxes, gt_labels,
            gt_bboxes_ignore=None, **kwargs):
        """Collect support-set SAR visual prototypes without running heads."""
        if gt_bboxes_ignore is not None:
            raise ValueError(
                'Ours2 visual prototype collection expects gt_bboxes_ignore=None.')
        img = self.gather_dict_values(img)
        self.gather_dict_values(img_metas)
        gt_bboxes = self.gather_dict_values(gt_bboxes, ignore_tensor=True)
        gt_labels = self.gather_dict_values(gt_labels, ignore_tensor=True)
        sar_images = img['sar']
        if not torch.is_tensor(sar_images) or sar_images.size(0) == 0:
            raise RuntimeError(
                'The visual prototype loader returned a batch without SAR images.')
        if img.get('rgb', []) or img.get('ifr', []):
            raise RuntimeError(
                'The visual prototype loader must contain SAR data only.')
        backbone_gt_bboxes = dict(sar=gt_bboxes['sar'], rgb=[], ifr=[])
        backbone_gt_labels = dict(sar=gt_labels['sar'], rgb=[], ifr=[])
        roi_features, text_features = (None, None)
        if self.enable_backbone_align_inputs:
            roi_features, text_features = self._collect_backbone_align_inputs(kwargs)
            if self.auto_backbone_text_from_aligner and text_features is None:
                text_features = self._build_backbone_text_features_from_aligner(
                    sar_images.device)
            roi_features = self._move_align_features_to_device(
                roi_features, sar_images.device)
            text_features = self._move_align_features_to_device(
                text_features, sar_images.device)
        self._forward_backbone(
            [sar_images], ['sar'],
            roi_features=roi_features,
            text_features=text_features,
            gt_bboxes=backbone_gt_bboxes,
            gt_labels=backbone_gt_labels)
        return {'loss_visual_prototype_collection': sar_images.new_zeros(())}

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, *args,
                      **kwargs):
        if self._routing_signature_collection_only:
            return self._collect_base_signature_batch(
                img,
                img_metas,
                gt_bboxes,
                gt_labels,
                *args,
                **kwargs)
        if self._novel_visual_prototype_collection_only:
            return self._collect_novel_visual_prototype_batch(
                img,
                img_metas,
                gt_bboxes,
                gt_labels,
                *args,
                **kwargs)
        self._routing_transfer_loss = None
        losses = super().forward_train(
            img, img_metas, gt_bboxes, gt_labels, *args, **kwargs)
        if (self._routing_transfer_loss is not None and
                isinstance(losses, dict)):
            losses['loss_b2n_routing_transfer'] = (
                self._routing_transfer_loss * self.routing_transfer_weight)
        return losses
