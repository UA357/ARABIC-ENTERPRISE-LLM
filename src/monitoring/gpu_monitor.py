import subprocess
import time


def get_gpu_stats():

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu="
            "utilization.gpu,"
            "memory.used,"
            "memory.total,"
            "temperature.gpu,"
            "power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )

    values = result.stdout.strip().split(",")

    return {
        "gpu_utilization_percent": float(values[0]),
        "memory_used_mb": float(values[1]),
        "memory_total_mb": float(values[2]),
        "temperature_c": float(values[3]),
        "power_w": float(values[4]),
    }


def main():

    print("=== GPU MONITOR ===")

    for i in range(5):

        stats = get_gpu_stats()

        print(
            f"[{i + 1}/5] "
            f"GPU: {stats['gpu_utilization_percent']:.0f}% | "
            f"VRAM: "
            f"{stats['memory_used_mb']:.0f}/"
            f"{stats['memory_total_mb']:.0f} MB | "
            f"Temp: "
            f"{stats['temperature_c']:.0f} C | "
            f"Power: "
            f"{stats['power_w']:.1f} W"
        )

        time.sleep(1)

    print("\n=== GPU MONITOR COMPLETE ===")


if __name__ == "__main__":
    main()