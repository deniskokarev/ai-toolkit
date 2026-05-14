import { NextResponse } from 'next/server';
import { exec, execSync } from 'child_process';
import { promisify } from 'util';
import { createRequire } from 'module';
import os from 'os';
import path from 'path';
import fs from 'fs';

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
      const spOut = execSync(
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

export async function GET() {
  try {
    const platform = os.platform();
    const isWindows = platform === 'win32';
    const isMac = platform === 'darwin';

    if (isMac) {
      const macGpu = await getMacGpuInfo();
      if (macGpu) {
        return NextResponse.json({
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
        });
      }
      return NextResponse.json({
        hasNvidiaSmi: false,
        isMac: true,
        gpus: [],
        error: 'Could not read Mac GPU stats',
      });
    }

    // NVIDIA first
    const hasNvidiaSmi = await checkNvidiaSmi(isWindows);
    if (hasNvidiaSmi) {
      const gpuStats = await getNvidiaGpuStats(isWindows);
      return NextResponse.json({
        hasNvidiaSmi: true,
        hasAmdSmi: false,
        hasRocmSmi: false,
        isMac: false,
        gpus: gpuStats,
      });
    }

    // AMD: prefer rocm-smi (more robust; amd-smi can crash on inactive iGPUs).
    const hasRocmSmi = await checkRocmSmi(isWindows);
    if (hasRocmSmi) {
      const gpuStats = await getRocmGpuStats(isWindows);
      return NextResponse.json({
        hasNvidiaSmi: false,
        hasAmdSmi: false,
        hasRocmSmi: true,
        isMac: false,
        gpus: gpuStats,
      });
    }

    // Fallback to amd-smi only if rocm-smi isn't present.
    const hasAmdSmi = await checkAmdSmi(isWindows);
    if (hasAmdSmi) {
      const gpuStats = await getAmdSmiGpuStats(isWindows);
      if (gpuStats && gpuStats.length > 0) {
        return NextResponse.json({
          hasNvidiaSmi: false,
          hasAmdSmi: true,
          hasRocmSmi: false,
          isMac: false,
          gpus: gpuStats,
        });
      }
    }

    return NextResponse.json({
      hasNvidiaSmi: false,
      hasAmdSmi: false,
      hasRocmSmi: false,
      isMac: false,
      gpus: [],
      error: 'Neither nvidia-smi, rocm-smi, nor amd-smi found. GPU detection unavailable.',
    });
  } catch (error) {
    console.error('Error fetching GPU stats:', error);
    return NextResponse.json(
      {
        hasNvidiaSmi: false,
        hasAmdSmi: false,
        hasRocmSmi: false,
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
      await execAsync('nvidia-smi -L');
    } else {
      await execAsync('which nvidia-smi');
    }
    return true;
  } catch {
    return false;
  }
}

async function checkRocmSmi(isWindows: boolean): Promise<boolean> {
  try {
    if (isWindows) {
      await execAsync('rocm-smi --version');
    } else {
      await execAsync('which rocm-smi');
    }
    return true;
  } catch {
    return false;
  }
}

async function checkAmdSmi(isWindows: boolean): Promise<boolean> {
  try {
    if (isWindows) {
      await execAsync('amd-smi --help');
    } else {
      await execAsync('which amd-smi');
    }
    return true;
  } catch {
    return false;
  }
}

async function getNvidiaGpuStats(isWindows: boolean) {
  const command =
    'nvidia-smi --query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,fan.speed --format=csv,noheader,nounits';

  const { stdout } = await execAsync(command, {
    env: { ...process.env, CUDA_DEVICE_ORDER: 'PCI_BUS_ID' },
  });

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

      const indexNum = parseInt(index) || 0;
      const tempNum = parseInt(temperature) || 0;
      const gpuUtilNum = parseInt(gpuUtil) || 0;
      const memoryUtilNum = parseInt(memoryUtil) || 0;
      const memoryTotalNum = parseInt(memoryTotal) || 0;
      const memoryFreeNum = parseInt(memoryFree) || 0;
      const memoryUsedNum = parseInt(memoryUsed) || 0;
      const powerDrawNum = parseFloat(powerDraw) || 0;
      const powerLimitNum = parseFloat(powerLimit) || 0;
      const clockGraphicsNum = parseInt(clockGraphics) || 0;
      const clockMemoryNum = parseInt(clockMemory) || 0;
      const fanSpeedNum = parseInt(fanSpeed) || 0;

      return {
        index: indexNum,
        name: name || `GPU ${indexNum}`,
        driverVersion: driverVersion || 'Unknown',
        temperature: Math.max(0, Math.min(200, tempNum)),
        utilization: {
          gpu: Math.max(0, Math.min(100, gpuUtilNum)),
          memory: Math.max(0, Math.min(100, memoryUtilNum)),
        },
        memory: {
          total: Math.max(0, memoryTotalNum),
          free: Math.max(0, Math.min(memoryTotalNum, memoryFreeNum)),
          used: Math.max(0, Math.min(memoryTotalNum, memoryUsedNum)),
        },
        power: {
          draw: Math.max(0, powerDrawNum),
          limit: Math.max(0, powerLimitNum),
        },
        clocks: {
          graphics: Math.max(0, clockGraphicsNum),
          memory: Math.max(0, clockMemoryNum),
        },
        fan: {
          speed: Math.max(0, Math.min(100, fanSpeedNum)),
        },
      };
    });

  return gpus;
}

// Prepend a project venv's bin directory to PATH so rocm-smi / amd-smi installed
// into the toolkit's venv are visible to the Node process.
function getVenvEnv(isWindows: boolean): NodeJS.ProcessEnv {
  const projectRoot = path.resolve(process.cwd(), '..');
  const env: NodeJS.ProcessEnv = { ...process.env };
  const venvBins = [
    path.join(projectRoot, '.venv', 'bin'),
    path.join(projectRoot, 'venv', 'bin'),
  ];
  for (const venvBin of venvBins) {
    if (fs.existsSync(venvBin)) {
      const sep = isWindows ? ';' : ':';
      env.PATH = `${venvBin}${sep}${process.env.PATH || ''}`;
      break;
    }
  }
  return env;
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

async function getRocmGpuStats(isWindows: boolean) {
  const command =
    'rocm-smi --showid --showproductname --showtemp --showuse --showmemuse --showmeminfo vram --showpower --showclocks --csv';

  try {
    const env = getVenvEnv(isWindows);
    const { stdout } = await execAsync(command, { env });

    const lines = stdout
      .split('\n')
      .map(l => l.trim())
      .filter(l => l.length > 0 && !l.startsWith('Exception') && !l.startsWith('Error'));

    const headerIndex = lines.findIndex(
      l => l.includes('device,GPU ID') || l.startsWith('device,'),
    );
    if (headerIndex === -1 || lines.length < headerIndex + 2) {
      return [];
    }

    const gpus = lines.slice(headerIndex + 1).map((line, idx) => {
      const fields = parseCSVLine(line);

      // rocm-smi CSV has two known layouts. Newer (~25 fields) puts Temperature at index 6;
      // older (~18 fields) puts it at index 2. Detect by field count.
      const isNewFormat = fields.length >= 25;
      while (fields.length < 25) fields.push('');

      const tempFieldIdx = isNewFormat ? 6 : 2;
      const mclkFieldIdx = isNewFormat ? 7 : 3;
      const sclkFieldIdx = isNewFormat ? 9 : 5;
      const powerFieldIdx = isNewFormat ? 13 : 9;
      const usageFieldIdx = isNewFormat ? 14 : 10;
      const memTotalFieldIdx = isNewFormat ? 17 : 12;
      const memUsedFieldIdx = isNewFormat ? 18 : 13;
      const cardSkuFieldIdx = isNewFormat ? 22 : 17;
      const cardModelFieldIdx = isNewFormat ? 20 : 15;
      const cardNameFieldIdx = isNewFormat ? 1 : -1;

      const deviceName = fields[0] || '';
      const deviceMatch = deviceName.match(/\d+/);
      const index = deviceMatch ? parseInt(deviceMatch[0]) : idx;

      // Temperature (°C)
      let temperature = 0;
      const tempStr = fields[tempFieldIdx] || '';
      const tempVal = parseRocmValue(tempStr);
      if (tempVal >= 0 && tempVal <= 200) temperature = tempVal;

      // GPU use (%)
      let gpuUtil = parseRocmValue(fields[usageFieldIdx]);
      gpuUtil = Math.max(0, Math.min(100, gpuUtil));

      // Memory (rocm-smi reports VRAM in bytes for this query)
      let memoryTotal = parseRocmValue(fields[memTotalFieldIdx]);
      let memoryUsed = parseRocmValue(fields[memUsedFieldIdx]);
      if (memoryTotal < 0 || isNaN(memoryTotal)) memoryTotal = 0;
      if (memoryUsed < 0 || isNaN(memoryUsed)) memoryUsed = 0;
      if (memoryUsed > memoryTotal) memoryUsed = memoryTotal;
      const memoryFree = Math.max(0, memoryTotal - memoryUsed);

      // Power (W): tolerate "(123.4 W)" / "123.45" / clock-shaped junk in same column
      let powerDraw = 0;
      const powerDrawStr = fields[powerFieldIdx] || '';
      if (powerDrawStr && !powerDrawStr.toLowerCase().includes('mhz') && powerDrawStr !== 'N/A') {
        const m = powerDrawStr.match(/(\d+\.?\d*)/);
        if (m) {
          const parsed = parseFloat(m[1]);
          if (parsed >= 0 && parsed <= 1000) powerDraw = parsed;
        }
      }

      // Clocks (MHz): "(1472Mhz)" or raw number
      const mclkMatch = (fields[mclkFieldIdx] || '').match(/(\d+)/);
      const sclkMatch = (fields[sclkFieldIdx] || '').match(/(\d+)/);
      let clockGraphics = sclkMatch ? parseInt(sclkMatch[1]) : 0;
      let clockMemory = mclkMatch ? parseInt(mclkMatch[1]) : 0;
      // Some rocm-smi builds report in Hz when it overflows — fold back to MHz.
      if (clockGraphics > 10000) clockGraphics = Math.round(clockGraphics / 1000000);
      if (clockMemory > 10000) clockMemory = Math.round(clockMemory / 1000000);
      if (clockGraphics < 0 || clockGraphics > 5000) clockGraphics = 0;
      if (clockMemory < 0 || clockMemory > 3000) clockMemory = 0;

      // Name: prefer Card SKU, then Card model, then Device Name, then fall back to "AMD GPU N".
      const cardSku = (fields[cardSkuFieldIdx] || '').trim();
      const cardModel = (fields[cardModelFieldIdx] || '').trim();
      const deviceNameField = cardNameFieldIdx >= 0 ? (fields[cardNameFieldIdx] || '').trim() : '';
      const cardVendor = (fields[16] || '').trim();
      const gpuId = (fields[1] || '').trim();

      let name = '';
      const looksLikeId = (s: string) => !s || s.startsWith('0x') || /^\d+$/.test(s);
      if (!looksLikeId(cardSku) && cardSku !== gpuId) {
        name = cardSku;
      } else if (!looksLikeId(cardModel) && cardModel !== gpuId) {
        name = cardModel;
      } else if (deviceNameField && !looksLikeId(deviceNameField)) {
        name = deviceNameField;
      } else if (cardVendor.includes('AMD') || cardVendor.includes('Advanced Micro Devices')) {
        name = `AMD GPU ${index}`;
      } else {
        name = `GPU ${index}`;
      }
      if (/^\d+$/.test(name)) name = `AMD GPU ${index}`;

      // Normalize memory units. rocm-smi reports VRAM in bytes for --showmeminfo, but
      // older / patched builds sometimes emit MB or GB. Detect by magnitude.
      let memoryTotalMB = 0, memoryUsedMB = 0, memoryFreeMB = 0;
      if (memoryTotal > 1024 * 1024 * 1024) {
        memoryTotalMB = Math.round(memoryTotal / (1024 * 1024));
        memoryUsedMB = Math.round(memoryUsed / (1024 * 1024));
        memoryFreeMB = Math.round(memoryFree / (1024 * 1024));
      } else if (memoryTotal > 1000) {
        memoryTotalMB = Math.round(memoryTotal);
        memoryUsedMB = Math.round(memoryUsed);
        memoryFreeMB = Math.round(memoryFree);
      } else if (memoryTotal > 0) {
        memoryTotalMB = Math.round(memoryTotal * 1024);
        memoryUsedMB = Math.round(memoryUsed * 1024);
        memoryFreeMB = Math.round(memoryFree * 1024);
      }

      const memoryUtilPercent = memoryTotalMB > 0
        ? Math.max(0, Math.min(100, Math.round((memoryUsedMB / memoryTotalMB) * 100)))
        : 0;

      return {
        index: isNaN(index) ? idx : index,
        name,
        driverVersion: 'ROCm',
        temperature: temperature > 0 ? Math.round(temperature) : 0,
        utilization: {
          gpu: Math.round(gpuUtil),
          memory: memoryUtilPercent,
        },
        memory: {
          total: memoryTotalMB,
          free: memoryFreeMB,
          used: memoryUsedMB,
        },
        power: {
          draw: Math.max(0, powerDraw),
          limit: 0,
        },
        clocks: {
          graphics: Math.max(0, clockGraphics),
          memory: Math.max(0, clockMemory),
        },
        fan: {
          speed: 0,
        },
      };
    });

    return gpus;
  } catch {
    // Last-ditch fallback: enumerate render nodes so the UI at least shows the cards.
    try {
      if (isWindows) return [];
      const { stdout } = await execAsync('ls -1 /dev/dri/renderD* 2>/dev/null | wc -l');
      const deviceCount = parseInt(stdout.trim()) || 0;
      return Array.from({ length: deviceCount }, (_, i) => ({
        index: i,
        name: `AMD GPU ${i}`,
        driverVersion: 'ROCm',
        temperature: 0,
        utilization: { gpu: 0, memory: 0 },
        memory: { total: 0, free: 0, used: 0 },
        power: { draw: 0, limit: 0 },
        clocks: { graphics: 0, memory: 0 },
        fan: { speed: 0 },
      }));
    } catch {
      return [];
    }
  }
}

async function getAmdSmiGpuStats(isWindows: boolean) {
  try {
    const env = getVenvEnv(isWindows);
    const { stdout: listStdout } = await execAsync('amd-smi list --json', { env });

    let gpuList: Array<{ gpu: number }> = [];
    try {
      const listData = JSON.parse(listStdout);
      if (Array.isArray(listData)) gpuList = listData;
      else if (listData && Array.isArray(listData.gpu_data)) gpuList = listData.gpu_data;
    } catch {
      return [];
    }
    if (gpuList.length === 0) return [];

    const gpus = await Promise.all(
      gpuList.map(async gpuInfo => {
        const gpuId = gpuInfo.gpu;
        try {
          const { stdout: metricStdout } = await execAsync(
            `amd-smi metric --gpu ${gpuId} --csv`,
            { env },
          );
          const lines = metricStdout.trim().split('\n').filter(l => l.trim().length > 0);
          if (lines.length < 2) return null;

          const header = lines[0].split(',');
          const dataLine = lines[1].split(',');

          const getFieldIndex = (fieldNames: string | string[]): number => {
            const names = Array.isArray(fieldNames) ? fieldNames : [fieldNames];
            for (const n of names) {
              const i = header.findIndex(h => h.toLowerCase().includes(n.toLowerCase()));
              if (i >= 0) return i;
            }
            return -1;
          };

          const gpuIndex = getFieldIndex('gpu');
          const usageIndex = getFieldIndex(['usage', 'gpu_use', 'utilization']);
          const edgeIndex = getFieldIndex(['edge', 'temperature', 'temp', 'junction']);
          const totalVramIndex = getFieldIndex(['total_vram', 'vram_total', 'memory_total']);
          const usedVramIndex = getFieldIndex(['used_vram', 'vram_used', 'memory_used']);
          const freeVramIndex = getFieldIndex(['free_vram', 'vram_free', 'memory_free']);
          const socketPowerIndex = getFieldIndex(['socket_power', 'power', 'power_draw', 'tdp']);
          const fanMaxIndex = getFieldIndex(['fan_max', 'fan_speed', 'max', 'fan_percent']);

          let gfxClkIndex = -1;
          for (let i = 0; i < header.length; i++) {
            const h = header[i].toLowerCase();
            if (h.startsWith('gfx_') && h.endsWith('_clk')) {
              gfxClkIndex = i;
              break;
            }
          }
          const memClkIndex = getFieldIndex('mem_0_clk');

          const index = gpuIndex >= 0 ? parseInt(dataLine[gpuIndex] || '0') || gpuId : gpuId;
          const usage = usageIndex >= 0 ? parseRocmValue(dataLine[usageIndex]) : 0;
          const temperature = edgeIndex >= 0 ? parseRocmValue(dataLine[edgeIndex]) : 0;
          const memoryTotalMB = totalVramIndex >= 0 ? parseRocmValue(dataLine[totalVramIndex]) : 0;
          const memoryUsedMB = usedVramIndex >= 0 ? parseRocmValue(dataLine[usedVramIndex]) : 0;
          const memoryFreeMB =
            freeVramIndex >= 0 ? parseRocmValue(dataLine[freeVramIndex]) : memoryTotalMB - memoryUsedMB;
          const powerDraw = socketPowerIndex >= 0 ? parseRocmValue(dataLine[socketPowerIndex]) : 0;
          const clockGraphics = gfxClkIndex >= 0 ? parseRocmValue(dataLine[gfxClkIndex]) : 0;
          const clockMemory = memClkIndex >= 0 ? parseRocmValue(dataLine[memClkIndex]) : 0;
          const fanSpeed =
            fanMaxIndex >= 0 && dataLine[fanMaxIndex] && dataLine[fanMaxIndex] !== 'N/A'
              ? parseRocmValue(dataLine[fanMaxIndex])
              : 0;

          let name = `AMD GPU ${index}`;
          try {
            const { stdout: staticStdout } = await execAsync(
              `amd-smi static --gpu ${gpuId} --json`,
              { env },
            );
            const staticData = JSON.parse(staticStdout);
            const gpuData = staticData?.gpu_data?.[0];
            if (gpuData) {
              if (gpuData.asic?.market_name) name = gpuData.asic.market_name;
              else if (gpuData.asic?.name) name = gpuData.asic.name;
              else if (gpuData.card?.market_name) name = gpuData.card.market_name;
              else if (gpuData.card?.name) name = gpuData.card.name;
              else if (gpuData.name) name = gpuData.name;
            }
          } catch {
            // keep default name
          }

          const validTemperature = temperature >= 0 && temperature <= 200 ? temperature : 0;
          const validUsage = Math.max(0, Math.min(100, usage));
          const validFanSpeed = Math.max(0, Math.min(100, fanSpeed));
          const memoryUtilPercent = memoryTotalMB > 0
            ? Math.max(0, Math.min(100, Math.round((memoryUsedMB / memoryTotalMB) * 100)))
            : 0;

          const hasBasicData = validTemperature > 0 || memoryTotalMB > 0;
          const hasPerformanceData =
            validUsage > 0 || powerDraw > 0 || clockGraphics > 0 || clockMemory > 0;
          const hasSufficientData = hasBasicData && hasPerformanceData;

          return {
            index,
            name,
            driverVersion: 'ROCm',
            temperature: validTemperature > 0 ? Math.round(validTemperature) : 0,
            utilization: { gpu: Math.round(validUsage), memory: memoryUtilPercent },
            memory: {
              total: Math.max(0, Math.round(memoryTotalMB)),
              free: Math.max(0, Math.round(memoryFreeMB)),
              used: Math.max(0, Math.round(memoryUsedMB)),
            },
            power: { draw: Math.max(0, powerDraw), limit: 0 },
            clocks: {
              graphics: Math.max(0, Math.round(clockGraphics)),
              memory: Math.max(0, Math.round(clockMemory)),
            },
            fan: { speed: validFanSpeed > 0 ? Math.round(validFanSpeed) : 0 },
            _hasSufficientData: hasSufficientData,
          };
        } catch {
          return null;
        }
      }),
    );

    const validGpus = gpus.filter((g): g is NonNullable<typeof g> => g !== null);
    const hasAnyData = validGpus.some(g => g._hasSufficientData);
    if (!hasAnyData && validGpus.length > 0) return [];
    return validGpus.map(({ _hasSufficientData, ...rest }) => rest);
  } catch {
    return [];
  }
}
