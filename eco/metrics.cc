#include "metrics.hpp"

float Metrics::center_error(const cv::Rect2f bbox, const cv::Rect2f bboxGroundtruth)
{
    float cx = bbox.x + bbox.width / 2.0f;
    float cy = bbox.y + bbox.height / 2.0f;
    float cx_gt = bboxGroundtruth.x + bboxGroundtruth.width / 2.0f;
    float cy_gt = bboxGroundtruth.y + bboxGroundtruth.height / 2.0f;
    float result = std::sqrt(std::pow((cx - cx_gt), 2) +
                             std::pow((cy - cy_gt), 2));
    return result;
}

float Metrics::iou(const cv::Rect2f bbox, const cv::Rect2f bboxGroundtruth)
{
    cv::Rect2f inter = Metrics::intersection(bbox, bboxGroundtruth);
    float area_bbox = bbox.area();
    float area_bbox_gt = bboxGroundtruth.area();
    float area_intersection = inter.area();
    float iou = area_bbox + area_bbox_gt - area_intersection;
    iou = area_intersection / (iou + 1e-12);
    return iou;
}

cv::Rect2f Metrics::intersection(const cv::Rect2f bbox,
                                 const cv::Rect2f bboxGroundtruth)
{
    float x1, y1, x2, y2, w, h;
    x1 = std::max(bbox.x, bboxGroundtruth.x);
    y1 = std::max(bbox.y, bboxGroundtruth.y);
    x2 = std::min(bbox.x + bbox.width, bboxGroundtruth.x + bboxGroundtruth.width);
    y2 = std::min(bbox.y + bbox.height, bboxGroundtruth.y + bboxGroundtruth.height);
    w = std::max(0.0f, x2 - x1);
    h = std::max(0.0f, y2 - y1);

    cv::Rect2f result(x1, y1, w, h);
    return result;
}

float Metrics::auc()
{
}