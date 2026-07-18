import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import { createRequire } from 'module';
import os from 'os';
import { cached } from '@/server/apiCache';

const execAsync = promisify(exec);

interface MacGpuResult {
  name: string;
  memUsed: number;
  memTotal: number;
  gpuLoad: number;
  temperature: number;
  fanSpeed: number;
  powerDraw: number;
}

async function getMacGpuInfo(): Promise<MacGpuResult | null> {
  try {
    const memoryTotal = os.totalmem() / (1024 * 1024);

    // Get GPU name and core count from system_profiler
    let gpuName = 'Apple GPU';
    try {
      const { stdout: spOut } = await execAsync(
        'system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset Model|Total Number of Cores"',
        { encoding: 'utf-8', timeout: 5000 },
      );
      const nameMatch = spOut.match(/Chipset Model:\s*(.+)/);
      const coresMatch = spOut.match(/Total Number of Cores:\s*(\d+)/);
      if (nameMatch) {
        gpuName = nameMatch[1].trim();
        if (coresMatch) {
          gpuName += ` GPU (${coresMatch[1]} cores)`;
        }
      }
    } catch {
      // fallback to generic name
    }

    let temperature = 0;
    let gpuLoad = 0;
    let fanSpeed = 0;
    let powerDraw = 0;
    let memUsed = 0;
    let memTotal = memoryTotal;

    try {
      // Use createRequire to hide from webpack static analysis so it doesn't fail on non-mac platforms
      const nativeRequire = createRequire(import.meta.url);
      const ms = nativeRequire('macstats') as any;

      try {
        const gpuData = ms.getGpuDataSync();
        temperature = gpuData.temperature || 0;
        gpuLoad = gpuData.usage || 0;
      } catch {
        // ignore
      }

      try {
        const fanData = ms.getFanDataSync();
        const fanKeys = Object.keys(fanData);
        if (fanKeys.length > 0) {
          fanSpeed = fanData[fanKeys[0]].rpm || 0;
        }
      } catch {
        // ignore
      }

      try {
        const powerData = ms.getPowerDataSync();
        powerDraw = powerData.gpu || 0;
      } catch {
        // ignore
      }

      try {
        const ramData = ms.getRAMUsageSync();
        memUsed = ramData.used / (1024 * 1024);
        memTotal = ramData.total / (1024 * 1024);
      } catch {
        // ignore
      }
    } catch (error) {
      console.warn('macstats not available:', error);
    }

    return { name: gpuName, memUsed, memTotal, gpuLoad, temperature, fanSpeed, powerDraw };
  } catch {
    return null;
  }
}

async function getGpuInfo() {
  // Get platform
  const platform = os.platform();
  const isWindows = platform === 'win32';
  const isMac = platform === 'darwin';

  if (isMac) {
    const macGpu = await getMacGpuInfo();
    if (macGpu) {
      return {
        hasNvidiaSmi: false,
        isMac: true,
        gpus: [
          {
            index: 0,
            name: macGpu.name,
            driverVersion: 'macOS',
            temperature: Math.round(macGpu.temperature),
            utilization: {
              gpu: macGpu.gpuLoad,
              memory: macGpu.memTotal > 0 ? Math.round((macGpu.memUsed / macGpu.memTotal) * 100) : 0,
            },
            memory: {
              total: Math.round(macGpu.memTotal),
              free: Math.round(macGpu.memTotal - macGpu.memUsed),
              used: Math.round(macGpu.memUsed),
            },
            power: { draw: macGpu.powerDraw, limit: 0 },
            clocks: { graphics: 0, memory: 0 },
            fan: { speed: macGpu.fanSpeed },
          },
        ],
      };
    }
    return {
      hasNvidiaSmi: false,
      isMac: true,
      gpus: [],
      error: 'Could not read Mac GPU stats',
    };
  }

  // Check if nvidia-smi is available
  const hasNvidiaSmi = await checkNvidiaSmi(isWindows);

  if (hasNvidiaSmi) {
    const gpuStats = await getGpuStats(isWindows);
    return {
      hasNvidiaSmi: true,
      gpus: gpuStats,
    };
  }

  // AMD ROCm fallback
  const hasRocmSmi = await checkRocmSmi(isWindows);
  if (hasRocmSmi) {
    const gpuStats = await getRocmGpuStats();
    return {
      hasNvidiaSmi: false,
      hasRocmSmi: true,
      isMac: false,
      gpus: gpuStats,
    };
  }

  return {
    hasNvidiaSmi: false,
    hasRocmSmi: false,
    isMac: false,
    gpus: [],
    error: 'Neither nvidia-smi nor rocm-smi found or accessible',
  };
}

export async function GET() {
  try {
    const gpuInfo = await cached('gpu-info', getGpuInfo);
    return NextResponse.json(gpuInfo);
  } catch (error) {
    console.error('Error fetching NVIDIA GPU stats:', error);
    return NextResponse.json(
      {
        hasNvidiaSmi: false,
        isMac: false,
        gpus: [],
        error: `Failed to fetch GPU stats: ${error instanceof Error ? error.message : String(error)}`,
      },
      { status: 500 },
    );
  }
}

async function checkNvidiaSmi(isWindows: boolean): Promise<boolean> {
  try {
    if (isWindows) {
      // Check if nvidia-smi is available on Windows
      // It's typically located in C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe
      // but we'll just try to run it directly as it may be in PATH
      await execAsync('nvidia-smi -L');
    } else {
      // Linux/macOS check
      await execAsync('which nvidia-smi');
    }
    return true;
  } catch (error) {
    return false;
  }
}

async function checkRocmSmi(isWindows: boolean): Promise<boolean> {
  if (isWindows) return false; // ROCm SMI is Linux-only
  try {
    await execAsync('which rocm-smi');
    return true;
  } catch {
    return false;
  }
}

function parseRocmValue(value: string | undefined, defaultValue: number = 0): number {
  if (!value || value === 'N/A' || value.trim() === '') return defaultValue;
  const parsed = parseFloat(value);
  return isNaN(parsed) ? defaultValue : parsed;
}

// Parse a single CSV line, respecting double-quoted fields.
function parseCSVLine(line: string): string[] {
  const fields: string[] = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (ch === ',' && !inQuotes) {
      fields.push(current.trim());
      current = '';
    } else {
      current += ch;
    }
  }
  fields.push(current.trim());
  return fields;
}

async function getRocmGpuStats() {
  const command =
    'rocm-smi --showid --showproductname --showtemp --showuse --showmemuse --showmeminfo vram --showpower --showclocks --csv';
  const { stdout } = await execAsync(command);

  const lines = stdout
    .split('\n')
    .map(l => l.trim())
    .filter(l => l.length > 0 && !l.startsWith('Exception') && !l.startsWith('Error'));

  const headerIndex = lines.findIndex(l => l.startsWith('device,'));
  if (headerIndex === -1 || lines.length < headerIndex + 2) {
    return [];
  }

  // Resolve column positions by header name -- rocm-smi's column layout
  // varies between ROCm versions, so hardcoded indices are not portable.
  const header = parseCSVLine(lines[headerIndex]).map(h => h.toLowerCase());
  const findCol = (...needles: string[]): number => {
    for (const needle of needles) {
      const n = needle.toLowerCase();
      const i = header.findIndex(h => h.includes(n));
      if (i >= 0) return i;
    }
    return -1;
  };

  const colDevice = findCol('device');
  const colDeviceName = header.findIndex(h => h === 'device name' || h.startsWith('device name'));
  const colTemp = findCol('temperature (sensor edge)', 'temperature');
  const colMclkSpeed = findCol('mclk clock speed');
  const colSclkSpeed = findCol('sclk clock speed');
  const colPower = findCol('graphics package power', 'power (w)');
  const colUsage = findCol('gpu use (%)', 'gpu use');
  const colVramTotal = findCol('vram total memory');
  const colVramUsed = findCol('vram total used memory');
  const colCardSeries = findCol('card series');
  const colCardModel = findCol('card model');
  const colCardVendor = findCol('card vendor');
  const colCardSku = findCol('card sku');

  return lines.slice(headerIndex + 1).map((line, idx) => {
    const fields = parseCSVLine(line);
    const get = (i: number): string => (i >= 0 && i < fields.length ? fields[i] : '');

    // Device index from "card0" / "card1" etc.
    const deviceMatch = get(colDevice).match(/\d+/);
    const index = deviceMatch ? parseInt(deviceMatch[0]) : idx;

    const tempVal = parseRocmValue(get(colTemp));
    const temperature = tempVal >= 0 && tempVal <= 200 ? Math.round(tempVal) : 0;

    const gpuUtil = Math.max(0, Math.min(100, parseRocmValue(get(colUsage))));

    // VRAM comes back in bytes for --showmeminfo; tolerate MB/GB from other builds.
    let memoryTotal = Math.max(0, parseRocmValue(get(colVramTotal)));
    let memoryUsed = Math.min(Math.max(0, parseRocmValue(get(colVramUsed))), memoryTotal);
    let toMB = (v: number) => v;
    if (memoryTotal > 1024 * 1024 * 1024) toMB = v => Math.round(v / (1024 * 1024));
    else if (memoryTotal <= 1000 && memoryTotal > 0) toMB = v => Math.round(v * 1024);
    const memoryTotalMB = toMB(memoryTotal);
    const memoryUsedMB = toMB(memoryUsed);

    // Power (W); the column sometimes carries clock strings on odd builds.
    let powerDraw = 0;
    const powerStr = get(colPower);
    if (powerStr && !powerStr.toLowerCase().includes('mhz')) {
      const m = powerStr.match(/(\d+\.?\d*)/);
      if (m) powerDraw = Math.min(1000, Math.max(0, parseFloat(m[1])));
    }

    // Clocks are formatted like "(1472Mhz)"
    const clockOf = (s: string, cap: number): number => {
      const m = s.match(/(\d+)/);
      let v = m ? parseInt(m[1]) : 0;
      if (v > 10000) v = Math.round(v / 1000000); // Hz -> MHz
      return v >= 0 && v <= cap ? v : 0;
    };

    // Name: prefer Device Name, then Card Series / Model / SKU; skip hex ids.
    const looksLikeId = (s: string) => !s || s.startsWith('0x') || /^\d+$/.test(s);
    const candidates = [get(colDeviceName), get(colCardSeries), get(colCardModel), get(colCardSku)];
    let name = candidates.find(c => c && !looksLikeId(c)) || '';
    if (!name) {
      const vendor = get(colCardVendor);
      name = vendor.includes('AMD') || vendor.includes('Advanced Micro Devices')
        ? `AMD GPU ${index}`
        : `GPU ${index}`;
    }

    return {
      index: isNaN(index) ? idx : index,
      name,
      driverVersion: 'ROCm',
      temperature,
      utilization: {
        gpu: Math.round(gpuUtil),
        memory: memoryTotalMB > 0 ? Math.round((memoryUsedMB / memoryTotalMB) * 100) : 0,
      },
      memory: {
        total: memoryTotalMB,
        free: Math.max(0, memoryTotalMB - memoryUsedMB),
        used: memoryUsedMB,
      },
      power: { draw: powerDraw, limit: 0 },
      clocks: {
        graphics: clockOf(get(colSclkSpeed), 5000),
        memory: clockOf(get(colMclkSpeed), 3000),
      },
      fan: { speed: 0 },
    };
  });
}

async function getGpuStats(isWindows: boolean) {
  // Command is the same for both platforms, but the path might be different
  const command =
    'nvidia-smi --query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,fan.speed --format=csv,noheader,nounits';

  // Execute command
  const { stdout } = await execAsync(command, {
    env: { ...process.env, CUDA_DEVICE_ORDER: 'PCI_BUS_ID' },
  });

  // Parse CSV output
  const gpus = stdout
    .trim()
    .split('\n')
    .map(line => {
      const [
        index,
        name,
        driverVersion,
        temperature,
        gpuUtil,
        memoryUtil,
        memoryTotal,
        memoryFree,
        memoryUsed,
        powerDraw,
        powerLimit,
        clockGraphics,
        clockMemory,
        fanSpeed,
      ] = line.split(', ').map(item => item.trim());

      return {
        index: parseInt(index),
        name,
        driverVersion,
        temperature: parseInt(temperature),
        utilization: {
          gpu: parseInt(gpuUtil),
          memory: parseInt(memoryUtil),
        },
        memory: {
          total: parseInt(memoryTotal),
          free: parseInt(memoryFree),
          used: parseInt(memoryUsed),
        },
        power: {
          draw: parseFloat(powerDraw),
          limit: parseFloat(powerLimit),
        },
        clocks: {
          graphics: parseInt(clockGraphics),
          memory: parseInt(clockMemory),
        },
        fan: {
          speed: parseInt(fanSpeed) || 0, // Some GPUs might not report fan speed, default to 0
        },
      };
    });

  return gpus;
}
