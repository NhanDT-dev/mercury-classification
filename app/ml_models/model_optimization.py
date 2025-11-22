"""Model optimization techniques including quantization, pruning, and distillation"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, Optional, Tuple, List
import numpy as np
from transformers import AutoModelForSequenceClassification
from app.utils.logger import logger


class ModelOptimizer:
    """
    Optimize models for production deployment.

    Techniques:
    - Quantization (INT8, FP16)
    - Pruning (unstructured, structured)
    - Knowledge distillation
    - ONNX export
    - TensorRT optimization
    """

    def __init__(self, model: nn.Module):
        """
        Initialize model optimizer.

        Args:
            model: PyTorch model to optimize
        """
        self.model = model
        self.original_size = self._get_model_size()

        logger.info(f"ModelOptimizer initialized. Original model size: {self.original_size:.2f} MB")

    def _get_model_size(self, model: Optional[nn.Module] = None) -> float:
        """
        Get model size in MB.

        Args:
            model: Model to measure (None = use self.model)

        Returns:
            Model size in megabytes
        """
        if model is None:
            model = self.model

        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024**2

        return size_mb

    def quantize_model(
        self,
        quantization_type: str = "dynamic",
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Quantize model to reduce size and improve inference speed.

        Args:
            quantization_type: "dynamic" or "static"
            dtype: Target data type (torch.qint8 or torch.float16)

        Returns:
            Quantized model
        """
        logger.info(f"Quantizing model using {quantization_type} quantization to {dtype}")

        model_quantized = self.model

        if quantization_type == "dynamic":
            # Dynamic quantization (weights only, activations computed on-the-fly)
            model_quantized = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear, nn.LSTM, nn.GRU},  # Layers to quantize
                dtype=dtype
            )

        elif quantization_type == "static":
            # Static quantization (requires calibration data)
            model_quantized = self.model
            model_quantized.eval()

            # Prepare model for quantization
            model_quantized.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model_quantized, inplace=True)

            # Calibrate (would need sample data here)
            # calibrate(model_quantized, calibration_data)

            # Convert to quantized model
            torch.quantization.convert(model_quantized, inplace=True)

        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")

        quantized_size = self._get_model_size(model_quantized)
        compression_ratio = self.original_size / quantized_size

        logger.info(
            f"Quantization complete. New size: {quantized_size:.2f} MB "
            f"(compression: {compression_ratio:.2f}x)"
        )

        return model_quantized

    def prune_model(
        self,
        pruning_amount: float = 0.3,
        pruning_type: str = "unstructured"
    ) -> nn.Module:
        """
        Prune model weights to reduce size.

        Args:
            pruning_amount: Fraction of weights to prune (0-1)
            pruning_type: "unstructured" or "structured"

        Returns:
            Pruned model
        """
        logger.info(f"Pruning {pruning_amount*100}% of weights using {pruning_type} pruning")

        model_pruned = self.model

        if pruning_type == "unstructured":
            # Global unstructured magnitude pruning
            parameters_to_prune = []

            for name, module in model_pruned.named_modules():
                if isinstance(module, nn.Linear):
                    parameters_to_prune.append((module, 'weight'))

            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_amount
            )

            # Make pruning permanent
            for module, param_name in parameters_to_prune:
                prune.remove(module, param_name)

        elif pruning_type == "structured":
            # Structured pruning (prune entire neurons/filters)
            for name, module in model_pruned.named_modules():
                if isinstance(module, nn.Linear):
                    prune.ln_structured(
                        module,
                        name='weight',
                        amount=pruning_amount,
                        n=2,  # L2 norm
                        dim=0  # Prune output neurons
                    )
                    prune.remove(module, 'weight')

        else:
            raise ValueError(f"Unknown pruning type: {pruning_type}")

        # Count remaining parameters
        total_params = sum(p.numel() for p in model_pruned.parameters())
        zero_params = sum((p == 0).sum().item() for p in model_pruned.parameters())
        sparsity = zero_params / total_params

        logger.info(f"Pruning complete. Sparsity: {sparsity*100:.2f}%")

        return model_pruned

    def convert_to_fp16(self) -> nn.Module:
        """
        Convert model to FP16 for faster inference.

        Returns:
            FP16 model
        """
        logger.info("Converting model to FP16")

        model_fp16 = self.model.half()

        fp16_size = self._get_model_size(model_fp16)
        compression = self.original_size / fp16_size

        logger.info(f"FP16 conversion complete. Size: {fp16_size:.2f} MB ({compression:.2f}x compression)")

        return model_fp16

    def get_optimization_report(
        self,
        optimized_model: nn.Module
    ) -> Dict:
        """
        Generate optimization report.

        Args:
            optimized_model: Optimized model

        Returns:
            Dict with optimization metrics
        """
        original_params = sum(p.numel() for p in self.model.parameters())
        optimized_params = sum(p.numel() for p in optimized_model.parameters())

        original_size = self.original_size
        optimized_size = self._get_model_size(optimized_model)

        # Count zero parameters (sparsity)
        zero_params = sum((p == 0).sum().item() for p in optimized_model.parameters())
        sparsity = zero_params / optimized_params if optimized_params > 0 else 0

        report = {
            "original": {
                "parameters": original_params,
                "size_mb": original_size
            },
            "optimized": {
                "parameters": optimized_params,
                "size_mb": optimized_size,
                "sparsity": sparsity
            },
            "compression": {
                "parameter_reduction": (original_params - optimized_params) / original_params,
                "size_reduction": (original_size - optimized_size) / original_size,
                "compression_ratio": original_size / optimized_size
            }
        }

        return report


class KnowledgeDistiller:
    """
    Knowledge distillation for model compression.

    Transfer knowledge from large teacher model to small student model.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 3.0,
        alpha: float = 0.5
    ):
        """
        Initialize knowledge distiller.

        Args:
            teacher_model: Large pre-trained teacher model
            student_model: Smaller student model to train
            temperature: Temperature for softening probabilities
            alpha: Weight for distillation loss (1-alpha for hard label loss)
        """
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha

        self.teacher.eval()  # Teacher in eval mode

        logger.info(
            f"KnowledgeDistiller initialized (T={temperature}, alpha={alpha})"
        )

    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute distillation loss.

        Args:
            student_logits: Student model outputs
            teacher_logits: Teacher model outputs
            labels: Optional true labels for hard loss

        Returns:
            Combined distillation loss
        """
        # Soft targets (distillation loss)
        student_soft = torch.nn.functional.log_softmax(
            student_logits / self.temperature, dim=1
        )
        teacher_soft = torch.nn.functional.softmax(
            teacher_logits / self.temperature, dim=1
        )

        # KL divergence loss (scaled by T^2)
        distillation_loss = torch.nn.functional.kl_div(
            student_soft, teacher_soft, reduction='batchmean'
        ) * (self.temperature ** 2)

        # Hard label loss (if labels provided)
        if labels is not None:
            hard_loss = torch.nn.functional.cross_entropy(student_logits, labels)
            total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        else:
            total_loss = distillation_loss

        return total_loss

    @torch.no_grad()
    def get_teacher_predictions(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Get soft targets from teacher.

        Args:
            inputs: Model inputs

        Returns:
            Teacher logits
        """
        teacher_outputs = self.teacher(**inputs)
        return teacher_outputs.logits


class ModelBenchmark:
    """
    Benchmark model performance (latency, throughput, memory).
    """

    @staticmethod
    @torch.no_grad()
    def measure_latency(
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        Measure model inference latency.

        Args:
            model: Model to benchmark
            input_ids: Sample input
            attention_mask: Attention mask
            num_runs: Number of inference runs
            warmup_runs: Warmup runs (not counted)

        Returns:
            Dict with latency statistics
        """
        import time

        model.eval()
        device = next(model.parameters()).device

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Warmup
        for _ in range(warmup_runs):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

        # Measure
        latencies = []
        for _ in range(num_runs):
            start = time.time()
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies.append((time.time() - start) * 1000)  # ms

        return {
            "mean_ms": np.mean(latencies),
            "std_ms": np.std(latencies),
            "min_ms": np.min(latencies),
            "max_ms": np.max(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99)
        }

    @staticmethod
    def measure_throughput(
        model: nn.Module,
        batch_sizes: List[int],
        seq_length: int = 128,
        num_runs: int = 50
    ) -> Dict[int, float]:
        """
        Measure throughput for different batch sizes.

        Args:
            model: Model to benchmark
            batch_sizes: List of batch sizes to test
            seq_length: Sequence length
            num_runs: Runs per batch size

        Returns:
            Dict mapping batch_size -> throughput (samples/sec)
        """
        import time

        model.eval()
        device = next(model.parameters()).device

        throughputs = {}

        for batch_size in batch_sizes:
            # Create dummy input
            input_ids = torch.randint(
                0, 1000, (batch_size, seq_length), device=device
            )
            attention_mask = torch.ones_like(input_ids)

            # Warmup
            for _ in range(10):
                _ = model(input_ids=input_ids, attention_mask=attention_mask)

            # Measure
            start = time.time()
            for _ in range(num_runs):
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
                if device.type == 'cuda':
                    torch.cuda.synchronize()

            elapsed = time.time() - start
            throughput = (batch_size * num_runs) / elapsed

            throughputs[batch_size] = throughput

            logger.info(f"Batch size {batch_size}: {throughput:.2f} samples/sec")

        return throughputs


# Utility functions
def optimize_for_inference(
    model: nn.Module,
    optimization_level: str = "medium"
) -> nn.Module:
    """
    Apply optimization pipeline for inference.

    Args:
        model: Model to optimize
        optimization_level: "light", "medium", or "aggressive"

    Returns:
        Optimized model
    """
    optimizer = ModelOptimizer(model)

    if optimization_level == "light":
        # FP16 only
        optimized = optimizer.convert_to_fp16()

    elif optimization_level == "medium":
        # FP16 + light pruning
        pruned = optimizer.prune_model(pruning_amount=0.2)
        optimized = ModelOptimizer(pruned).convert_to_fp16()

    elif optimization_level == "aggressive":
        # Pruning + quantization
        pruned = optimizer.prune_model(pruning_amount=0.4)
        optimized = ModelOptimizer(pruned).quantize_model()

    else:
        raise ValueError(f"Unknown optimization level: {optimization_level}")

    # Generate report
    report = optimizer.get_optimization_report(optimized)
    logger.info(f"Optimization report: {report}")

    return optimized
