from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "visioninspect-api"
    vision_engine: str = "opencv"


class InspectionParameters(BaseModel):
    canny_low: int = Field(default=50, ge=10, le=200)
    canny_high: int = Field(default=150, ge=50, le=400)
    min_defect_area: int = Field(default=50, ge=10, le=2000)
    pass_fail_threshold: float = Field(default=1.0, ge=0.1, le=20.0)
    blur_kernel: int = Field(default=5)


class InspectionMetrics(BaseModel):
    num_defects: int
    total_defect_area: float
    image_area: int
    defect_percentage: float
    confidence: float


class InspectionImages(BaseModel):
    original: str
    grayscale: str
    edges: str
    annotated: str


class InspectionResponse(BaseModel):
    inspection_id: str
    filename: str
    image_width: int
    image_height: int
    inspection_timestamp: str
    inspection_duration_ms: int
    parameters: InspectionParameters
    metrics: InspectionMetrics
    verdict: str
    images: InspectionImages


class SampleImage(BaseModel):
    filename: str
    category: str
    preview_url: str


class SamplesResponse(BaseModel):
    samples: list[SampleImage]


class MetricsResponse(BaseModel):
    total_inspections: int
    pass_count: int
    fail_count: int
    pass_rate: float
    average_defect_coverage: float
    average_inspection_time_ms: float

