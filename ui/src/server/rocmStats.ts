import { execFile } from 'child_process';
import { promisify } from 'util';
import { GpuInfo } from '@/types';

const execFileAsync = promisify(execFile);

/**
 * AMD GPU stats via rocm-smi, shared by the always-on monitor
 * (src/server/monitor.ts) and the legacy /api/gpu route.
 *
 * Unlike nvidia-smi there is no loop mode, so every sample costs a process
 * spawn (~150-350 ms at 99% of a core on ROCm 7.x -- rocm-smi is a Python
 * script). Callers must not sample this at the monitor tick cadence; see
 * ROCM_REFRESH_MS in monitor.ts.
 */

const ROCM_ARGS = [
  '--showid',
  '--showproductname',
  '--showtemp',
  '--showuse',
  '--showmemuse',
  '--showmeminfo',
  'vram',
  '--showpower',
  '--showclocks',
  '--csv',
];

export async function checkRocmSmi(isWindows: boolean): Promise<boolean> {
  if (isWindows) return false; // ROCm SMI is Linux-only
  try {
    await execFileAsync('which', ['rocm-smi']);
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

export function parseRocmCsv(stdout: string): GpuInfo[] {
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
    const memoryTotal = Math.max(0, parseRocmValue(get(colVramTotal)));
    const memoryUsed = Math.min(Math.max(0, parseRocmValue(get(colVramUsed))), memoryTotal);
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
      name = vendor.includes('AMD') || vendor.includes('Advanced Micro Devices') ? `AMD GPU ${index}` : `GPU ${index}`;
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

export async function getRocmGpuStats(): Promise<GpuInfo[]> {
  const { stdout } = await execFileAsync('rocm-smi', ROCM_ARGS);
  return parseRocmCsv(stdout);
}
