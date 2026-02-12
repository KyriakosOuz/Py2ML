const { execSync } = require('child_process');
const path = require('path');

const tsNode = 'npx ts-node --compiler-options \'{"module":"CommonJS"}\'';
const dir = path.resolve(__dirname);

const scripts = [
  'seed.ts',
  'seed-stage4.ts',
  'seed-stage5.ts',
  'seed-stage6.ts',
  'seed-stage7.ts',
  'seed-stage8.ts',
];

console.log('Running all seeds (each skips if already seeded)...\n');

for (const script of scripts) {
  const full = path.join(dir, script);
  try {
    execSync(`${tsNode} "${full}"`, { stdio: 'inherit', cwd: path.resolve(dir, '..') });
  } catch (e) {
    console.error(`Warning: ${script} failed, continuing...`);
  }
}

console.log('\nAll seeds complete.');
