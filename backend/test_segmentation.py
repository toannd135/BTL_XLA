"""Regression checks for mask fidelity, instance identity and conservative refinement."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from detection import (compute_mask_metrics, detect_seg, simplify_polygon, split_banana_cluster_detections,
                       split_fruit_cluster_detections, deduplicate_masks, _crop_fruit_instance, _fruit_proposals)
from quality import assess_detection


class Values:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def tolist(self):
        return self.values.tolist()

    def __len__(self):
        return len(self.values)


class Obbs(SimpleNamespace):
    def __len__(self):
        return len(self.cls)


class Model:
    def __init__(self, result, names=None, task="segment"):
        self.result = result
        self.names = names or {0: "banana", 1: "apple", 2: "orange"}
        self.task = task
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


def rectangle(x1, y1, x2, y2):
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def raster(polygon, shape=(100, 220)):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 1)
    return mask


def sam_result(masks, scores=None):
    return SimpleNamespace(masks=SimpleNamespace(data=Values(masks)), boxes=SimpleNamespace(
        cls=Values(range(len(masks))), conf=Values(scores or [0.95]*len(masks))))


class SegmentationTests(unittest.TestCase):
    def test_mask_dedup_removes_identical_same_class_masks(self):
        polygons = [rectangle(10, 10, 60, 60), rectangle(11, 11, 59, 59),
                    rectangle(55, 10, 105, 60)]
        boxes = np.asarray([[10, 10, 60, 60], [11, 11, 59, 59], [55, 10, 105, 60]], np.float32)
        output = deduplicate_masks(polygons, boxes, [.9, .8, .7], [47, 47, 47], 120, 80)
        self.assertEqual(len(output[0]), 2)

    def test_mask_dedup_keeps_touching_masks_with_overlapping_boxes(self):
        polygons = [rectangle(10, 10, 60, 60), rectangle(55, 10, 105, 60)]
        boxes = np.asarray([[10, 10, 65, 60], [50, 10, 105, 60]], np.float32)
        output = deduplicate_masks(polygons, boxes, [.9, .8], [47, 47], 120, 80)
        self.assertEqual(len(output[0]), 2)

    def test_native_contour_keeps_narrow_tip_and_concavity(self):
        polygon = np.asarray([[10, 40], [80, 40], [81, 5], [82, 40], [110, 40],
                              [110, 80], [70, 80], [70, 60], [40, 60], [40, 80], [10, 80]], dtype=np.float32)
        result = SimpleNamespace(boxes=Obbs(xyxy=Values([[10, 5, 110, 80]]),
                                  conf=Values([0.9]), cls=Values([46])),
                                 masks=SimpleNamespace(xy=[polygon]))
        names = {i: str(i) for i in range(80)}
        names.update({46: "banana", 47: "apple", 49: "orange"})
        model = Model(result, names=names)
        polys, boxes, scores, classes, metrics = detect_seg(model, np.zeros((100, 220, 3), np.uint8))
        self.assertTrue(model.calls[0]["retina_masks"])
        self.assertEqual(classes.tolist(), [46])
        self.assertEqual(float(polys[0][:, 1].min()), 5)
        np.testing.assert_array_equal(raster(polys[0]), raster(polygon))
        self.assertTrue(metrics[0]["mask_valid"])

    def test_missing_mask_is_not_replaced_with_a_fake_rectangle(self):
        result = SimpleNamespace(boxes=Obbs(xyxy=Values([[10, 10, 30, 30], [50, 10, 70, 30]]),
                                  conf=Values([0.9, 0.8]), cls=Values([1, 2])),
                                 masks=SimpleNamespace(xy=[rectangle(10, 10, 30, 30)]))
        polygons, _, _, _, metrics = detect_seg(Model(result), np.zeros((100, 100, 3), np.uint8))
        self.assertEqual(len(polygons), 2)
        self.assertEqual(len(polygons[1]), 0)
        self.assertFalse(metrics[1]["mask_valid"])

    def test_simplification_cannot_remove_a_thin_object(self):
        polygon = rectangle(1, 1, 100, 1.1)
        self.assertGreaterEqual(len(simplify_polygon(polygon, epsilon=2)), 3)

    def test_roi_metrics_equal_full_image_union_even_with_stale_boxes(self):
        polygons = [rectangle(30, 20, 90, 80), rectangle(30, 20, 50, 80), rectangle(70, 20, 90, 80)]
        masks = [raster(p) for p in polygons]
        metrics = compute_mask_metrics(polygons, np.zeros((3, 4)), 220, 100)
        expected = float((masks[0] & (masks[1] | masks[2])).sum()) / float(masks[0].sum())
        self.assertEqual(metrics[0]["overlap_ratio"], round(expected, 3))
        self.assertEqual(metrics[0]["mask_area"], int(masks[0].sum()))
        self.assertEqual([d["other_idx"] for d in metrics[0]["overlaps"]], [1, 2])

    def test_overlapping_boxes_with_separate_masks_do_not_trigger_duplicate_warning(self):
        polygons = [np.asarray([[10, 10], [90, 10], [10, 90]], np.float32),
                    np.asarray([[90, 90], [90, 12], [12, 90]], np.float32)]
        boxes = [[10, 10, 90, 90]] * 2
        metrics = compute_mask_metrics(polygons, boxes, 300, 300)
        objects = [{"label": "banana", "bbox": box, "confidence": 0.9, **metric,
                    "mask_overlaps": metric["overlaps"]} for box, metric in zip(boxes, metrics)]
        self.assertEqual(assess_detection(objects, 300, 300)["status"], "ready")

    def test_identical_masks_still_trigger_review(self):
        polygons = [rectangle(10, 10, 40, 40)] * 2
        metrics = compute_mask_metrics(polygons, [[10, 10, 40, 40]]*2, 300, 300)
        objects = [{"label": "apple", "bbox": [10, 10, 40, 40], "confidence": 0.9,
                    **metric, "mask_overlaps": metric["overlaps"]} for metric in metrics]
        quality = assess_detection(objects, 300, 300)
        self.assertEqual(quality["status"], "needs_review")
        self.assertTrue(any("IoU 100%" in reason for reason in quality["reasons"]))

    def test_one_fruit_disagreement_and_new_split_require_review(self):
        quality = assess_detection([], 100, 100, {"primary_task": "segment", "primary_count": 1, "fallback_count": 2})
        self.assertEqual(quality["status"], "needs_review")
        quality = assess_detection([], 100, 100, {"banana_split_groups": 1})
        self.assertEqual(quality["status"], "needs_review")


class BananaRefinementTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 220, 3), np.uint8)
        self.parent = rectangle(10, 10, 209, 89)
        self.children = [rectangle(10, 10, 109, 89), rectangle(110, 10, 209, 89)]
        self.obb = Model(SimpleNamespace(obb=Obbs(cls=np.asarray([0, 0]), conf=np.asarray([0.85, 0.8]),
                         xyxyxyxy=[Values(p) for p in self.children])))

    def split(self, sam=None, polygons=None, classes=None):
        polygons = polygons if polygons is not None else [self.parent]
        classes = classes if classes is not None else [46]
        boxes = np.asarray([np.concatenate((p.min(axis=0), p.max(axis=0))) for p in polygons], np.float32)
        diagnostics = {}
        output = split_banana_cluster_detections(polygons, boxes, np.full(len(polygons), .7, np.float32),
                   np.asarray(classes), self.obb, self.image, det_names={46: "banana", 47: "apple"},
                   mask_model=sam, diagnostics=diagnostics)
        return output, diagnostics

    def test_missing_refiner_keeps_parent_and_marks_review(self):
        output, diagnostics = self.split()
        self.assertEqual(len(output[1]), 1)
        np.testing.assert_array_equal(output[0][0], self.parent)
        self.assertEqual(diagnostics["banana_split_pending"], 1)

    def test_valid_sam_masks_keep_coco_class_and_conservative_confidence(self):
        sam = Model(sam_result([raster(p) for p in self.children]))
        output, diagnostics = self.split(sam)
        self.assertEqual(len(output[1]), 2)
        self.assertEqual(output[3].tolist(), [46, 46])
        self.assertTrue(np.all(output[2] <= .7))
        self.assertEqual(diagnostics["banana_split_groups"], 1)
        self.assertEqual(len(sam.calls), 1)
        self.assertEqual(sam.calls[0]["labels"], [[1, 0, 0, 0]]*2)
        for polygon, expected in zip(output[0], self.children):
            np.testing.assert_array_equal(raster(polygon), raster(expected))

    def test_duplicate_or_incomplete_sam_replacement_is_atomic(self):
        scenarios = [[raster(self.parent)]*2, [raster(self.children[0])],
                     [raster(self.children[0]), raster(rectangle(180, 50, 182, 52))]]
        for masks in scenarios:
            with self.subTest(masks=len(masks)):
                output, diagnostics = self.split(Model(sam_result(masks)))
                self.assertEqual(len(output[1]), 1)
                np.testing.assert_array_equal(output[0][0], self.parent)
                self.assertEqual(diagnostics["banana_split_pending"], 1)

    def test_sam_mask_must_agree_with_its_proposal(self):
        masks = [raster(self.children[0]), raster(rectangle(170, 10, 209, 89))]
        output, diagnostics = self.split(Model(sam_result(masks)))
        self.assertEqual(len(output[1]), 1)
        np.testing.assert_array_equal(raster(output[0][0]), raster(self.parent))
        self.assertEqual(diagnostics["banana_split_pending"], 1)

    def test_low_quality_sam_is_rejected(self):
        output, diagnostics = self.split(Model(sam_result([raster(p) for p in self.children], [.95, .3])))
        self.assertEqual(len(output[1]), 1)
        self.assertEqual(diagnostics["banana_split_pending"], 1)

    def test_each_obb_belongs_to_only_one_existing_instance(self):
        sam = Model(sam_result([raster(p) for p in self.children]))
        output, diagnostics = self.split(sam, self.children, [46, 46])
        self.assertEqual(len(output[1]), 2)
        self.assertEqual(sam.calls, [])
        self.assertEqual(diagnostics["banana_split_pending"], 0)

    def test_non_banana_image_skips_helper_inference(self):
        output, _ = self.split(polygons=[self.parent], classes=[47])
        self.assertEqual(len(output[1]), 1)
        self.assertEqual(self.obb.calls, [])

    def test_runtime_refiner_failure_preserves_prediction(self):
        sam = Model(None)
        with patch.object(sam, "predict", side_effect=RuntimeError("out of memory")):
            output, diagnostics = self.split(sam)
        self.assertEqual(len(output[1]), 1)
        self.assertEqual(diagnostics["banana_refine_error"], "RuntimeError")


class AllFruitRefinementTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 220, 3), np.uint8)
        self.parent = rectangle(10, 10, 209, 89)
        self.children = [rectangle(10, 10, 109, 89), rectangle(110, 10, 209, 89)]
        self.names = {46:"banana",47:"apple",49:"orange"}

    def obb(self, label):
        cls = {"banana":0,"apple":1,"orange":2}[label]
        return Model(SimpleNamespace(obb=Obbs(cls=np.asarray([cls, cls]), conf=np.asarray([.85,.8]),
                     xyxyxyxy=[Values(p) for p in self.children])))

    def refine(self, label="apple", obb=True, hbb=None, sam=None, seg=None, polygons=None):
        polygons = polygons if polygons is not None else [self.parent]
        cls = {"banana":46,"apple":47,"orange":49}[label]
        boxes = np.asarray([np.concatenate((p.min(axis=0),p.max(axis=0))) for p in polygons],np.float32).reshape(-1,4)
        diagnostics = {}
        result = split_fruit_cluster_detections(polygons, boxes, np.full(len(polygons),.9),
            np.full(len(polygons),cls), self.obb(label) if obb else None,self.image,
            det_names=self.names, mask_model=sam, seg_model=seg, diagnostics=diagnostics,
            hbb_detections=hbb,hbb_names={0:"banana",1:"apple",2:"orange"})
        return result,diagnostics

    def test_apples_and_oranges_are_split_and_keep_coco_class_ids(self):
        for label,cls in (("apple",47),("orange",49)):
            with self.subTest(label=label):
                result,diag = self.refine(label=label,sam=Model(sam_result([raster(p) for p in self.children])))
                self.assertEqual(len(result[1]),2)
                self.assertEqual(result[3].tolist(),[cls,cls])
                self.assertEqual(diag["instance_refinement"][label]["split_groups"],1)
                self.assertEqual(diag["banana_split_groups"],0)

    def test_hbb_can_propose_separate_apples_without_an_obb_model(self):
        hbb = (np.asarray([[10,10,109,89],[110,10,209,89]],np.float32),np.asarray([.9,.85]),np.asarray([1,1]))
        result,diag = self.refine(obb=False,hbb=hbb,sam=Model(sam_result([raster(p) for p in self.children])))
        self.assertEqual(len(result[1]),2)
        self.assertEqual(diag["fruit_split_groups"],1)

    def test_hbb_group_cannot_override_separate_obb_bananas(self):
        hbb = (np.asarray([[10,10,209,89]],np.float32),np.asarray([.99]),np.asarray([0]))
        result,diag = self.refine(label="banana",hbb=hbb,
                                 sam=Model(sam_result([raster(p) for p in self.children])))
        self.assertEqual(len(result[1]),2)
        self.assertEqual(diag["banana_split_groups"],1)

    def test_group_proposal_with_four_children_does_not_hide_individual_hypotheses(self):
        boxes = np.asarray([[10,10,209,89],[10,10,59,89],[60,10,109,89],
                            [110,10,159,89],[160,10,209,89]],np.float32)
        detections = (boxes,np.asarray([.95,.9,.9,.9,.9]),np.asarray([1]*5))
        proposals = _fruit_proposals(None,self.image,{"apple"},.4,None,detections,{1:"apple"})
        self.assertEqual(len(proposals),4)
        self.assertTrue(all(proposal["area"] < 5000 for proposal in proposals))

    def test_obb_and_hbb_agreeing_on_one_fruit_are_not_two_instances(self):
        hbb = (np.asarray([[10,10,109,89]],np.float32),np.asarray([.95]),np.asarray([1]))
        sam = Model(None)
        result,_ = self.refine(polygons=[self.children[0]],hbb=hbb,sam=sam)
        self.assertEqual(len(result[1]),1)
        self.assertEqual(sam.calls,[])

    def test_crop_verification_recovers_an_apple_missing_from_native_masks(self):
        with patch("detection._crop_fruit_instance",return_value=(self.children[1],.93)) as verify:
            result,diag = self.refine(polygons=[self.children[0]],seg=object())
        self.assertEqual(len(result[1]),2)
        self.assertEqual(result[3].tolist(),[47,47])
        self.assertAlmostEqual(result[2][1],.93,places=5)
        self.assertEqual(diag["fruit_recovered"],1)
        self.assertEqual(verify.call_count,1)

    def test_an_auxiliary_box_is_not_added_without_a_semantic_mask(self):
        with patch("detection._crop_fruit_instance",return_value=None):
            result,diag = self.refine(polygons=[self.children[0]],seg=object())
        self.assertEqual(len(result[1]),1)
        self.assertEqual(diag["fruit_recovered"],0)

    def test_optional_crop_failure_does_not_fail_primary_predictions(self):
        with patch("detection._crop_fruit_instance",side_effect=RuntimeError("out of memory")):
            result,diag = self.refine(polygons=[self.children[0]],seg=object())
        self.assertEqual(len(result[1]),1)
        self.assertEqual(diag["fruit_refine_error"],"RuntimeError")

    def test_a_crop_showing_the_existing_fruit_cannot_duplicate_it(self):
        with patch("detection._crop_fruit_instance",return_value=(self.children[0],.99)):
            result,diag = self.refine(polygons=[self.children[0]],seg=object())
        self.assertEqual(len(result[1]),1)
        self.assertEqual(diag["fruit_recovered"],0)

    def test_tiny_high_confidence_grape_proposal_is_not_recovered_as_apple(self):
        hbb = (np.asarray([[150,30,160,40]],np.float32),np.asarray([.99]),np.asarray([1]))
        with patch("detection._crop_fruit_instance") as verify:
            result,diag = self.refine(obb=False,hbb=hbb,polygons=[self.children[0]],seg=object())
        self.assertEqual(len(result[1]),1)
        self.assertEqual(diag["fruit_recovered"],0)
        verify.assert_not_called()

    def test_empty_scene_does_not_admit_auxiliary_false_positives(self):
        hbb = (np.asarray([[10,10,109,89]],np.float32),np.asarray([.99]),np.asarray([1]))
        with patch("detection._crop_fruit_instance") as verify:
            result,_ = self.refine(obb=False,hbb=hbb,polygons=[],seg=object())
        self.assertEqual(len(result[1]),0)
        verify.assert_not_called()

    def test_all_children_can_be_verified_by_crops_when_sam_cannot_split(self):
        with patch("detection._crop_fruit_instance",side_effect=[(p,.9) for p in self.children]):
            result,diag = self.refine(seg=object())
        self.assertEqual(len(result[1]),2)
        self.assertEqual(diag["fruit_split_groups"],1)

    def test_partial_crop_replacement_cannot_delete_the_parent(self):
        with patch("detection._crop_fruit_instance",side_effect=[(self.children[0],.9),None]):
            result,diag = self.refine(seg=object())
        self.assertEqual(len(result[1]),1)
        np.testing.assert_array_equal(result[0][0],self.parent)
        self.assertEqual(diag["instance_refinement"]["apple"]["pending_groups"],1)

    def test_crop_returns_original_image_coordinates_and_rejects_wrong_class(self):
        proposal = {"pts":rectangle(20,20,70,70),"mask":raster(rectangle(20,20,70,70)),"area":2601}
        result = SimpleNamespace(boxes=Obbs(xyxy=Values([[15,15,65,65]]),conf=Values([.9]),cls=Values([1])),
                                 masks=SimpleNamespace(xy=[rectangle(15,15,65,65)]))
        model = Model(result)
        verified = _crop_fruit_instance(model,self.image,proposal,"apple",.4,None)
        self.assertIsNotNone(verified)
        np.testing.assert_array_equal(verified[0],rectangle(20,20,70,70))
        self.assertIsNone(_crop_fruit_instance(model,self.image,proposal,"orange",.4,None))

    def test_crop_boundary_cannot_cut_one_fruit_into_a_new_instance(self):
        proposal = {"pts":rectangle(20,20,70,70),"mask":raster(rectangle(20,20,70,70)),"area":2601}
        result = SimpleNamespace(boxes=Obbs(xyxy=Values([[0,15,50,65]]),conf=Values([.9]),cls=Values([1])),
                                 masks=SimpleNamespace(xy=[rectangle(0,15,50,65)]))
        self.assertIsNone(_crop_fruit_instance(Model(result),self.image,proposal,"apple",.4,None))

    def test_new_class_requires_hbb_support_and_preserves_detector_class_mapping(self):
        hbb = (np.asarray([[110,10,209,89]],np.float32),np.asarray([.9]),np.asarray([2]))
        with patch("detection._crop_fruit_instance",return_value=(self.children[1],.9)):
            result,diag = self.refine(obb=False,hbb=hbb,polygons=[self.children[0]],seg=object())
        self.assertEqual(result[3].tolist(),[47,49])
        self.assertEqual(diag["instance_refinement"]["orange"]["recovered"],1)

    def test_quality_warning_names_apples_instead_of_bananas(self):
        quality = assess_detection([],100,100,{"instance_refinement":{"apple":{"pending_groups":1}}})
        self.assertEqual(quality["status"],"needs_review")
        self.assertTrue(any("táo" in reason for reason in quality["reasons"]))
        self.assertFalse(any("chuối" in reason for reason in quality["reasons"]))

    def test_pending_banana_disables_automatic_count(self):
        quality = assess_detection([], 100, 100, {"banana_split_pending": 1})
        self.assertEqual(quality["decision"], "do_not_auto_count")
        self.assertFalse(quality["automatic_count_allowed"])


class PipelineTests(unittest.TestCase):
    def test_upload_preserves_original_resolution_by_default(self):
        import asyncio
        import io
        from starlette.datastructures import Headers, UploadFile
        import main
        image = np.full((100, 150, 3), 128, np.uint8)
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)
        upload = UploadFile(io.BytesIO(encoded.tobytes()), filename="small.png",
                            headers=Headers({"content-type": "image/png"}))
        with patch.object(main, "ENHANCE_IMAGE", False):
            _, decoded = asyncio.run(main.read_image(upload))
        np.testing.assert_array_equal(decoded, image)

    def test_clip_filtering_recomputes_overlap_indices(self):
        import main
        names = {i: str(i) for i in range(80)}
        names.update({46: "banana", 47: "apple", 49: "orange"})
        result = SimpleNamespace(boxes=Obbs(xyxy=Values([[10, 10, 40, 40]]*2),
                                  conf=Values([.95, .95]), cls=Values([47, 47])),
                                 masks=SimpleNamespace(xy=[rectangle(10, 10, 40, 40)]*2))
        classifier = SimpleNamespace(names={0: "fresh"}, predict=lambda crops, **kw: [
            SimpleNamespace(probs=SimpleNamespace(top1=0, top1conf=.99)) for crop in crops])
        clip = SimpleNamespace(names={0: "apple", 1: "banana", 2: "orange", 3: "other"},
                               probs_boxes=lambda *args, **kw: [[.99, 0, 0, .01], [0, 0, 0, 1]])
        with patch.dict(main.models, {"det": Model(result, names), "cls": classifier, "clip": clip}, clear=True):
            objects, diagnostics = main.predict(np.zeros((100, 100, 3), np.uint8))
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["overlap_ratio"], 0)
        self.assertEqual(objects[0]["mask_overlaps"], [])
        self.assertEqual(objects[0]["geometry_type"], "mask")
        self.assertEqual(diagnostics["primary_count"], 1)


if __name__ == "__main__":
    unittest.main()
