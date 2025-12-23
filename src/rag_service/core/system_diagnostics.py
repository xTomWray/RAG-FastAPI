"""Comprehensive system diagnostics for debugging crashes during heavy workloads.

This module monitors the full system state to help diagnose shutdowns and crashes
that occur during computationally intensive tasks like GPU embedding operations.

Key areas monitored:
1. CPU - temperature, frequency, utilization, throttling
2. GPU - covered by gpu_diagnostics.py
3. Memory (RAM) - usage, pressure, errors
4. Storage - I/O, temperature, available space
5. Power - system power draw, voltage rails (if available)
6. Motherboard - VRM temps, chipset temps, fan speeds (via hardware sensors)
7. Process - thread count, memory allocations, file descriptors
8. Event Logs - recent critical system events (Windows)

Common crash causes and their diagnostic signatures:
- Thermal shutdown: High temps shortly before crash, throttling detected
- PSU overload: High power draw, voltage rail drops (if measurable)
- VRM overheat: Motherboard sensor temps high (needs hardware monitor)
- Memory errors: RAM errors in logs, high memory pressure
- Driver crash: Events in Windows Event Log, BSOD codes
"""

import datetime
import os
import platform
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO


@dataclass
class CPUDiagnostics:
    """CPU diagnostic information."""

    # Basic info
    cpu_name: str = ""
    physical_cores: int = 0
    logical_cores: int = 0

    # Utilization
    total_utilization: float = 0.0
    per_core_utilization: list[float] = field(default_factory=list)

    # Frequency
    current_freq_mhz: float = 0.0
    min_freq_mhz: float = 0.0
    max_freq_mhz: float = 0.0
    per_core_freq_mhz: list[float] = field(default_factory=list)

    # Temperature (if available)
    temperature_c: float | None = None
    per_core_temps_c: list[float] = field(default_factory=list)

    # Load averages (Unix) or queue length (Windows)
    load_avg_1m: float | None = None
    load_avg_5m: float | None = None
    load_avg_15m: float | None = None

    # Context switches and interrupts
    context_switches: int | None = None
    interrupts: int | None = None

    def to_log_lines(self) -> list[str]:
        """Convert to log lines."""
        lines = []

        # Utilization
        util_line = f"    CPU_UTIL: {self.total_utilization:.1f}%"
        if self.per_core_utilization:
            max_core = max(self.per_core_utilization)
            util_line += f" (max_core: {max_core:.1f}%)"
        lines.append(util_line)

        # Frequency
        if self.current_freq_mhz > 0:
            freq_line = f"    CPU_FREQ: {self.current_freq_mhz:.0f}MHz"
            if self.max_freq_mhz > 0:
                freq_pct = (self.current_freq_mhz / self.max_freq_mhz) * 100
                freq_line += f"/{self.max_freq_mhz:.0f}MHz ({freq_pct:.0f}%)"
            lines.append(freq_line)

        # Temperature
        if self.temperature_c is not None:
            temp_line = f"    CPU_TEMP: {self.temperature_c:.1f}°C"
            if self.per_core_temps_c:
                max_temp = max(self.per_core_temps_c)
                temp_line += f" (max_core: {max_temp:.1f}°C)"
            lines.append(temp_line)

        return lines


@dataclass
class MemoryDiagnostics:
    """Memory (RAM) diagnostic information."""

    # Physical memory
    total_gb: float = 0.0
    used_gb: float = 0.0
    available_gb: float = 0.0
    percent_used: float = 0.0

    # Swap/Page file
    swap_total_gb: float = 0.0
    swap_used_gb: float = 0.0
    swap_percent: float = 0.0

    # Memory pressure indicators
    page_faults: int | None = None
    commit_charge_gb: float | None = None
    commit_limit_gb: float | None = None

    # Process-specific
    process_memory_mb: float = 0.0
    process_memory_percent: float = 0.0

    def to_log_lines(self) -> list[str]:
        """Convert to log lines."""
        lines = []

        # Physical RAM
        lines.append(
            f"    RAM: {self.used_gb:.1f}/{self.total_gb:.1f}GB "
            f"({self.percent_used:.1f}%) | avail: {self.available_gb:.1f}GB"
        )

        # Swap
        if self.swap_total_gb > 0:
            lines.append(
                f"    SWAP: {self.swap_used_gb:.1f}/{self.swap_total_gb:.1f}GB "
                f"({self.swap_percent:.1f}%)"
            )

        # Process memory
        if self.process_memory_mb > 0:
            lines.append(
                f"    PROC_MEM: {self.process_memory_mb:.1f}MB "
                f"({self.process_memory_percent:.1f}% of system)"
            )

        return lines


@dataclass
class DiskDiagnostics:
    """Disk I/O diagnostic information."""

    # Space
    disk_path: str = "."
    total_gb: float = 0.0
    used_gb: float = 0.0
    free_gb: float = 0.0
    percent_used: float = 0.0

    # I/O counters
    read_bytes_sec: float | None = None
    write_bytes_sec: float | None = None
    read_count: int | None = None
    write_count: int | None = None

    # I/O wait (indicates disk bottleneck)
    io_wait_percent: float | None = None

    def to_log_lines(self) -> list[str]:
        """Convert to log lines."""
        lines = []

        lines.append(
            f"    DISK: {self.used_gb:.1f}/{self.total_gb:.1f}GB "
            f"({self.percent_used:.1f}%) | free: {self.free_gb:.1f}GB"
        )

        if self.read_bytes_sec is not None:
            io_line = f"    DISK_IO: read={self.read_bytes_sec/1024/1024:.1f}MB/s"
            if self.write_bytes_sec is not None:
                io_line += f", write={self.write_bytes_sec/1024/1024:.1f}MB/s"
            lines.append(io_line)

        return lines


@dataclass
class HardwareSensorData:
    """Hardware sensor data from motherboard/system sensors.

    Note: Requires OpenHardwareMonitor, HWiNFO, or LibreHardwareMonitor
    running with shared memory enabled for Windows.
    """

    # VRM (Voltage Regulator Module) - critical for power delivery
    vrm_temp_c: float | None = None

    # Chipset
    chipset_temp_c: float | None = None

    # Motherboard
    motherboard_temp_c: float | None = None

    # Voltages (if available)
    voltage_12v: float | None = None
    voltage_5v: float | None = None
    voltage_3_3v: float | None = None
    voltage_vcore: float | None = None

    # Fan speeds
    cpu_fan_rpm: int | None = None
    case_fan_rpms: list[int] = field(default_factory=list)

    # Power (from motherboard sensors or UPS)
    system_power_w: float | None = None

    def to_log_lines(self) -> list[str]:
        """Convert to log lines."""
        lines = []

        # Temperatures
        temps = []
        if self.vrm_temp_c is not None:
            temps.append(f"VRM={self.vrm_temp_c:.0f}°C")
        if self.chipset_temp_c is not None:
            temps.append(f"Chipset={self.chipset_temp_c:.0f}°C")
        if self.motherboard_temp_c is not None:
            temps.append(f"MB={self.motherboard_temp_c:.0f}°C")

        if temps:
            lines.append(f"    HW_TEMPS: {', '.join(temps)}")

        # Voltages
        volts = []
        if self.voltage_12v is not None:
            volts.append(f"12V={self.voltage_12v:.2f}V")
        if self.voltage_5v is not None:
            volts.append(f"5V={self.voltage_5v:.2f}V")
        if self.voltage_vcore is not None:
            volts.append(f"Vcore={self.voltage_vcore:.3f}V")

        if volts:
            lines.append(f"    HW_VOLTS: {', '.join(volts)}")

        # Power
        if self.system_power_w is not None:
            lines.append(f"    SYS_PWR: {self.system_power_w:.0f}W")

        # Fans
        if self.cpu_fan_rpm is not None:
            fan_line = f"    FANS: CPU={self.cpu_fan_rpm}RPM"
            if self.case_fan_rpms:
                fan_line += f", Case={self.case_fan_rpms}"
            lines.append(fan_line)

        return lines


@dataclass
class ProcessDiagnostics:
    """Current process diagnostic information."""

    pid: int = 0
    thread_count: int = 0
    handle_count: int | None = None  # Windows file handles
    fd_count: int | None = None  # Unix file descriptors

    # Memory details
    rss_mb: float = 0.0  # Resident Set Size
    vms_mb: float = 0.0  # Virtual Memory Size

    # CPU
    cpu_percent: float = 0.0
    cpu_times_user: float = 0.0
    cpu_times_system: float = 0.0

    def to_log_lines(self) -> list[str]:
        """Convert to log lines."""
        lines = []

        proc_line = f"    PROC: pid={self.pid}, threads={self.thread_count}"
        if self.handle_count:
            proc_line += f", handles={self.handle_count}"
        proc_line += f", cpu={self.cpu_percent:.1f}%"
        lines.append(proc_line)

        lines.append(f"    PROC_MEM: RSS={self.rss_mb:.1f}MB, VMS={self.vms_mb:.1f}MB")

        return lines


@dataclass
class WindowsEventInfo:
    """Recent Windows Event Log entries."""

    critical_events: list[dict] = field(default_factory=list)
    error_events: list[dict] = field(default_factory=list)
    warning_events: list[dict] = field(default_factory=list)

    # BSOD info (if recent)
    last_bsod_code: str | None = None
    last_bsod_time: str | None = None
    last_bsod_driver: str | None = None

    def to_log_lines(self) -> list[str]:
        """Convert to log lines."""
        lines = []

        if self.last_bsod_code:
            lines.append(f"    LAST_BSOD: {self.last_bsod_code} at {self.last_bsod_time}")
            if self.last_bsod_driver:
                lines.append(f"    BSOD_DRIVER: {self.last_bsod_driver}")

        if self.critical_events:
            lines.append(f"    CRIT_EVENTS: {len(self.critical_events)} in last hour")
            for event in self.critical_events[:3]:
                lines.append(f"      - {event.get('source', 'Unknown')}: {event.get('message', '')[:80]}")

        return lines


@dataclass
class SystemDiagnostics:
    """Complete system diagnostic snapshot."""

    timestamp: str = ""
    hostname: str = ""
    os_info: str = ""
    uptime_hours: float | None = None

    cpu: CPUDiagnostics = field(default_factory=CPUDiagnostics)
    memory: MemoryDiagnostics = field(default_factory=MemoryDiagnostics)
    disk: DiskDiagnostics = field(default_factory=DiskDiagnostics)
    hardware: HardwareSensorData = field(default_factory=HardwareSensorData)
    process: ProcessDiagnostics = field(default_factory=ProcessDiagnostics)
    windows_events: WindowsEventInfo = field(default_factory=WindowsEventInfo)

    def to_log_lines(self, include_all: bool = False) -> list[str]:
        """Convert to log lines.

        Args:
            include_all: If True, include all diagnostics. If False, only critical ones.
        """
        lines = []

        # CPU is critical for thermal issues
        lines.extend(self.cpu.to_log_lines())

        # Memory
        lines.extend(self.memory.to_log_lines())

        # Hardware sensors are critical for diagnosing power/thermal issues
        hw_lines = self.hardware.to_log_lines()
        if hw_lines:
            lines.extend(hw_lines)

        if include_all:
            # Disk
            lines.extend(self.disk.to_log_lines())

            # Process
            lines.extend(self.process.to_log_lines())

            # Windows events
            event_lines = self.windows_events.to_log_lines()
            if event_lines:
                lines.extend(event_lines)

        return lines


def collect_cpu_diagnostics() -> CPUDiagnostics:
    """Collect CPU diagnostic information."""
    diag = CPUDiagnostics()

    try:
        import psutil

        # Basic info
        diag.physical_cores = psutil.cpu_count(logical=False) or 0
        diag.logical_cores = psutil.cpu_count(logical=True) or 0

        # Utilization
        diag.total_utilization = psutil.cpu_percent(interval=None)
        try:
            diag.per_core_utilization = psutil.cpu_percent(interval=None, percpu=True)
        except Exception:
            pass

        # Frequency
        try:
            freq = psutil.cpu_freq()
            if freq:
                diag.current_freq_mhz = freq.current
                diag.min_freq_mhz = freq.min
                diag.max_freq_mhz = freq.max
        except Exception:
            pass

        # Per-core frequency (if available)
        try:
            per_cpu_freq = psutil.cpu_freq(percpu=True)
            if per_cpu_freq:
                diag.per_core_freq_mhz = [f.current for f in per_cpu_freq]
        except Exception:
            pass

        # Load average (Unix) or approximate on Windows
        try:
            load = psutil.getloadavg()
            diag.load_avg_1m = load[0]
            diag.load_avg_5m = load[1]
            diag.load_avg_15m = load[2]
        except (AttributeError, OSError):
            pass

        # Context switches and interrupts
        try:
            stats = psutil.cpu_stats()
            diag.context_switches = stats.ctx_switches
            diag.interrupts = stats.interrupts
        except Exception:
            pass

    except ImportError:
        pass

    # CPU temperature via different methods
    diag.temperature_c = _get_cpu_temperature()

    # CPU name
    diag.cpu_name = _get_cpu_name()

    return diag


def _get_cpu_name() -> str:
    """Get CPU name/model."""
    try:
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            return cpu_name.strip()
        else:
            # Linux
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor()


def _get_cpu_temperature() -> float | None:
    """Get CPU temperature using various methods."""

    # Method 1: psutil sensors (Linux primarily)
    try:
        import psutil
        temps = psutil.sensors_temperatures()
        if temps:
            # Look for CPU temp in common sensor names
            for name in ['coretemp', 'cpu_thermal', 'k10temp', 'zenpower']:
                if name in temps:
                    core_temps = [t.current for t in temps[name] if t.current > 0]
                    if core_temps:
                        return max(core_temps)
    except (ImportError, AttributeError):
        pass

    # Method 2: WMI on Windows (requires admin or specific drivers)
    if platform.system() == "Windows":
        try:
            import wmi
            w = wmi.WMI(namespace="root\\wmi")
            temp_info = w.MSAcpi_ThermalZoneTemperature()
            if temp_info:
                # Convert from tenths of Kelvin to Celsius
                temp_k = temp_info[0].CurrentTemperature / 10.0
                return temp_k - 273.15
        except Exception:
            pass

        # Try OpenHardwareMonitor WMI
        try:
            import wmi
            w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
            sensors = w.Sensor()
            for sensor in sensors:
                if sensor.SensorType == "Temperature" and "CPU" in sensor.Name:
                    return float(sensor.Value)
        except Exception:
            pass

    return None


def collect_memory_diagnostics() -> MemoryDiagnostics:
    """Collect memory diagnostic information."""
    diag = MemoryDiagnostics()

    try:
        import psutil

        # Physical memory
        mem = psutil.virtual_memory()
        diag.total_gb = mem.total / (1024**3)
        diag.used_gb = mem.used / (1024**3)
        diag.available_gb = mem.available / (1024**3)
        diag.percent_used = mem.percent

        # Swap
        swap = psutil.swap_memory()
        diag.swap_total_gb = swap.total / (1024**3)
        diag.swap_used_gb = swap.used / (1024**3)
        diag.swap_percent = swap.percent

        # Process memory
        process = psutil.Process()
        mem_info = process.memory_info()
        diag.process_memory_mb = mem_info.rss / (1024**2)
        diag.process_memory_percent = process.memory_percent()

    except ImportError:
        pass

    return diag


def collect_disk_diagnostics(path: str = ".") -> DiskDiagnostics:
    """Collect disk diagnostic information."""
    diag = DiskDiagnostics(disk_path=path)

    try:
        import psutil

        # Disk usage
        usage = psutil.disk_usage(path)
        diag.total_gb = usage.total / (1024**3)
        diag.used_gb = usage.used / (1024**3)
        diag.free_gb = usage.free / (1024**3)
        diag.percent_used = usage.percent

        # I/O counters (system-wide)
        try:
            io = psutil.disk_io_counters()
            if io:
                diag.read_count = io.read_count
                diag.write_count = io.write_count
        except Exception:
            pass

    except ImportError:
        pass

    return diag


def collect_process_diagnostics() -> ProcessDiagnostics:
    """Collect current process diagnostic information."""
    diag = ProcessDiagnostics()

    try:
        import psutil

        process = psutil.Process()
        diag.pid = process.pid
        diag.thread_count = process.num_threads()

        # Memory
        mem_info = process.memory_info()
        diag.rss_mb = mem_info.rss / (1024**2)
        diag.vms_mb = mem_info.vms / (1024**2)

        # CPU
        diag.cpu_percent = process.cpu_percent()
        cpu_times = process.cpu_times()
        diag.cpu_times_user = cpu_times.user
        diag.cpu_times_system = cpu_times.system

        # Handles/FDs
        if platform.system() == "Windows":
            try:
                diag.handle_count = process.num_handles()
            except Exception:
                pass
        else:
            try:
                diag.fd_count = process.num_fds()
            except Exception:
                pass

    except ImportError:
        pass

    return diag


def collect_hardware_sensors() -> HardwareSensorData:
    """Collect hardware sensor data.

    Note: This requires additional software on Windows:
    - OpenHardwareMonitor with WMI enabled, or
    - LibreHardwareMonitor with WMI enabled, or
    - HWiNFO with shared memory enabled
    """
    data = HardwareSensorData()

    if platform.system() != "Windows":
        return data

    # Try OpenHardwareMonitor WMI interface
    try:
        import wmi
        w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
        sensors = w.Sensor()

        for sensor in sensors:
            name = sensor.Name.lower()
            value = float(sensor.Value) if sensor.Value else None

            if value is None:
                continue

            if sensor.SensorType == "Temperature":
                if "vrm" in name or "vr " in name:
                    data.vrm_temp_c = value
                elif "chipset" in name:
                    data.chipset_temp_c = value
                elif "motherboard" in name or "system" in name:
                    data.motherboard_temp_c = value

            elif sensor.SensorType == "Voltage":
                if "+12v" in name or "12v" in name:
                    data.voltage_12v = value
                elif "+5v" in name or "5v" in name:
                    data.voltage_5v = value
                elif "+3.3v" in name or "3.3v" in name:
                    data.voltage_3_3v = value
                elif "vcore" in name:
                    data.voltage_vcore = value

            elif sensor.SensorType == "Fan":
                if "cpu" in name:
                    data.cpu_fan_rpm = int(value)
                else:
                    data.case_fan_rpms.append(int(value))

            elif sensor.SensorType == "Power":
                if "cpu package" in name or "cpu total" in name:
                    # CPU power, not system power
                    pass
                elif "system" in name or "total" in name:
                    data.system_power_w = value

    except Exception:
        pass

    # Try LibreHardwareMonitor (similar WMI interface)
    try:
        import wmi
        w = wmi.WMI(namespace="root\\LibreHardwareMonitor")
        # Similar logic as above...
    except Exception:
        pass

    return data


def get_windows_events(hours_back: int = 1) -> WindowsEventInfo:
    """Get recent Windows Event Log entries.

    Args:
        hours_back: How many hours back to search.
    """
    info = WindowsEventInfo()

    if platform.system() != "Windows":
        return info

    try:
        import win32evtlog
        import win32evtlogutil

        server = None
        log_type = "System"

        hand = win32evtlog.OpenEventLog(server, log_type)

        # Calculate time threshold
        now = datetime.datetime.now()
        threshold = now - datetime.timedelta(hours=hours_back)

        flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ

        events = win32evtlog.ReadEventLog(hand, flags, 0)

        while events:
            for event in events:
                event_time = datetime.datetime.strptime(
                    str(event.TimeGenerated), "%m/%d/%y %H:%M:%S"
                )

                if event_time < threshold:
                    break

                event_dict = {
                    "time": str(event.TimeGenerated),
                    "source": event.SourceName,
                    "event_id": event.EventID,
                    "message": win32evtlogutil.SafeFormatMessage(event, log_type)[:200]
                }

                if event.EventType == win32evtlog.EVENTLOG_ERROR_TYPE:
                    info.error_events.append(event_dict)
                elif event.EventType == win32evtlog.EVENTLOG_WARNING_TYPE:
                    info.warning_events.append(event_dict)

                # Check for BSOD (BugCheck)
                if "BugCheck" in event.SourceName or event.EventID == 1001:
                    info.last_bsod_time = str(event.TimeGenerated)
                    # Parse BSOD code from message if available
                    msg = event_dict.get("message", "")
                    if "0x" in msg:
                        # Extract hex code
                        import re
                        match = re.search(r"0x[0-9A-Fa-f]+", msg)
                        if match:
                            info.last_bsod_code = match.group()

            events = win32evtlog.ReadEventLog(hand, flags, 0)

        win32evtlog.CloseEventLog(hand)

    except ImportError:
        # pywin32 not installed, try PowerShell
        try:
            result = subprocess.run(
                [
                    "powershell", "-Command",
                    f"Get-EventLog -LogName System -EntryType Error,Warning "
                    f"-After (Get-Date).AddHours(-{hours_back}) -Newest 10 | "
                    "Select-Object TimeGenerated,Source,EventID,Message | ConvertTo-Json"
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout:
                import json
                events = json.loads(result.stdout)
                if isinstance(events, dict):
                    events = [events]
                for event in events:
                    info.error_events.append({
                        "time": event.get("TimeGenerated", ""),
                        "source": event.get("Source", ""),
                        "message": str(event.get("Message", ""))[:200]
                    })
        except Exception:
            pass
    except Exception:
        pass

    return info


def collect_system_diagnostics() -> SystemDiagnostics:
    """Collect complete system diagnostics."""
    diag = SystemDiagnostics(
        timestamp=datetime.datetime.now().isoformat(timespec="milliseconds"),
        hostname=platform.node(),
        os_info=f"{platform.system()} {platform.release()}"
    )

    # Uptime
    try:
        import psutil
        boot_time = psutil.boot_time()
        uptime_seconds = datetime.datetime.now().timestamp() - boot_time
        diag.uptime_hours = uptime_seconds / 3600
    except Exception:
        pass

    # Collect all subsystem diagnostics
    diag.cpu = collect_cpu_diagnostics()
    diag.memory = collect_memory_diagnostics()
    diag.disk = collect_disk_diagnostics()
    diag.hardware = collect_hardware_sensors()
    diag.process = collect_process_diagnostics()
    diag.windows_events = get_windows_events(hours_back=1)

    return diag


class SystemDiagnosticLogger:
    """Logger for system diagnostics with crash-safe writing."""

    def __init__(
        self,
        log_dir: Path | str = "./logs",
        log_name: str = "system_diag",
        enabled: bool = True,
    ):
        self._enabled = enabled
        self._lock = threading.Lock()
        self._file: TextIO | None = None
        self._log_path: Path | None = None

        if enabled:
            self._setup_log_file(Path(log_dir), log_name)

    def _setup_log_file(self, log_dir: Path, log_name: str) -> None:
        """Create log file."""
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_path = log_dir / f"{log_name}_{date_str}.log"
            self._file = open(self._log_path, "a", encoding="utf-8", buffering=1)
            self._write_header()
        except Exception as e:
            print(f"WARNING: Could not create system diagnostic log: {e}", file=sys.stderr)
            self._enabled = False

    def _write_header(self) -> None:
        """Write log header."""
        if not self._file:
            return

        diag = collect_system_diagnostics()

        lines = [
            "",
            "=" * 100,
            f"SYSTEM DIAGNOSTIC LOG - {datetime.datetime.now().isoformat()}",
            "=" * 100,
            f"Hostname: {diag.hostname}",
            f"OS: {diag.os_info}",
            f"CPU: {diag.cpu.cpu_name}",
            f"Cores: {diag.cpu.physical_cores} physical, {diag.cpu.logical_cores} logical",
            f"RAM: {diag.memory.total_gb:.1f} GB",
            f"Uptime: {diag.uptime_hours:.1f} hours" if diag.uptime_hours else "",
            "=" * 100,
            "",
        ]

        for line in lines:
            if line:
                self._file.write(line + "\n")
        self._flush()

    def _flush(self) -> None:
        """Flush to disk."""
        if self._file:
            self._file.flush()
            try:
                os.fsync(self._file.fileno())
            except Exception:
                pass

    def log_state(self, label: str = "CHECKPOINT", include_all: bool = False) -> SystemDiagnostics:
        """Log current system state."""
        if not self._enabled or not self._file:
            return collect_system_diagnostics()

        with self._lock:
            diag = collect_system_diagnostics()

            timestamp = datetime.datetime.now().isoformat(timespec="milliseconds")
            self._file.write(f"[{timestamp}] {label}\n")

            for line in diag.to_log_lines(include_all=include_all):
                self._file.write(line + "\n")

            self._flush()
            return diag

    def close(self) -> None:
        """Close log file."""
        if self._file:
            try:
                self.log_state("SESSION_END", include_all=True)
                self._file.close()
            except Exception:
                pass
            self._file = None


# Global instance
_system_logger: SystemDiagnosticLogger | None = None
_logger_lock = threading.Lock()


def get_system_logger(
    log_dir: Path | str = "./logs",
    enabled: bool = True,
) -> SystemDiagnosticLogger:
    """Get or create the global system diagnostic logger."""
    global _system_logger
    with _logger_lock:
        if _system_logger is None:
            _system_logger = SystemDiagnosticLogger(log_dir=log_dir, enabled=enabled)
    return _system_logger
