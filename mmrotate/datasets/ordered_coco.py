"""Keep COCO label IDs independent of JSON category ordering."""


def ordered_category_ids(coco, classes):
    categories = coco.load_cats(coco.get_cat_ids())
    by_name = {}
    for category in categories:
        by_name.setdefault(category['name'], []).append(category['id'])
    ids = []
    for name in classes:
        matches = by_name.get(name, [])
        if len(matches) != 1:
            raise ValueError(f'Expected one COCO category named {name!r}, got {matches}.')
        ids.append(matches[0])
    return ids


class OrderedCocoLabels:
    def load_annotations(self, ann_file):
        infos = super().load_annotations(ann_file)
        self.cat_ids = ordered_category_ids(self.coco, self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        return infos

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        # CocoDataset.evaluate resets cat_ids in JSON order before exporting.
        # Restore label order here so class-indexed predictions use the right IDs.
        self.cat_ids = ordered_category_ids(self.coco, self.CLASSES)
        return super().format_results(results, jsonfile_prefix, **kwargs)

    def evaluate_det_segm(self, *args, **kwargs):
        # COCOeval sorts category IDs internally; match that order for the
        # per-class AP table after predictions have been exported in label order.
        label_ids = self.cat_ids
        self.cat_ids = sorted(label_ids)
        try:
            return super().evaluate_det_segm(*args, **kwargs)
        finally:
            self.cat_ids = label_ids
