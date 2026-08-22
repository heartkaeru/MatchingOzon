"""Бенчмарк инференса: замер latency, throughput и потребления памяти (GPU/RAM)."""

import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Бенчмарк инференса модели: latency / throughput / память."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Путь к данным для инференса",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Размер батча (по умолчанию: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Устройство инференса (по умолчанию: автоопределение)",
    )
    return parser.parse_args()


def resolve_device(requested):
    if requested is not None:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        print("[предупреждение] torch не установлен, используется cpu")
        return "cpu"


def load_data(data_path):
    # TODO: загрузка данных для инференса из data_path
    raise NotImplementedError


def run_inference(model, batch, device):
    # TODO: прогон одного батча через модель на device
    raise NotImplementedError


def measure_latency(model, data, device, batch_size, n_warmup=5, n_iters=50):
    """Среднее время инференса одного батча, сек."""
    # TODO: n_warmup прогревочных итераций, затем замер n_iters через time.perf_counter()
    raise NotImplementedError


def measure_throughput(model, data, device, batch_size):
    """Пропускная способность, примеров/сек."""
    # TODO: общее число обработанных примеров / суммарное время
    raise NotImplementedError


def report_memory(device):
    """Пиковое потребление памяти: GPU (torch.cuda) и RAM (psutil), МБ."""
    stats = {"gpu_peak_mb": None, "ram_peak_mb": None}
    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            stats["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    except ImportError:
        pass
    try:
        import psutil

        stats["ram_peak_mb"] = psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        pass
    return stats


def print_report(results):
    print("\n===== Отчет о бенчмарке =====")
    for key, value in results.items():
        formatted = f"{value:.2f}" if isinstance(value, float) else str(value)
        print(f"{key:>15}: {formatted}")
    print("=============================")


def main():
    start = time.perf_counter()
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[инфо] device={device}, batch_size={args.batch_size}, data_path={args.data_path}")

    results = {
        "device": device,
        "batch_size": args.batch_size,
        "latency_ms": None,
        "throughput_sps": None,
        **report_memory(device),
        "total_time_s": time.perf_counter() - start,
    }

    # TODO: полный пайплайн бенчмарка:
    #   data = load_data(args.data_path)
    #   model = <загрузка модели>
    #   results["latency_ms"] = measure_latency(model, data, device, args.batch_size) * 1000
    #   results["throughput_sps"] = measure_throughput(model, data, device, args.batch_size)

    print_report(results)


if __name__ == "__main__":
    main()

