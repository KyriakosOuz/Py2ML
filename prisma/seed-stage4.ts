export {};

const { PrismaClient } = require('@prisma/client');
const { seedModule6 } = require('./seed-stage4-module6');
const { seedModule7 } = require('./seed-stage4-module7');

const prisma = new PrismaClient();

async function main() {
  const existing = await prisma.stage.findUnique({ where: { id: 'stage-004' } });
  if (existing) { console.log('Stage 4 already seeded. Skipping.'); return; }

  console.log('═══════════════════════════════════════════');
  console.log('  Seeding Stage 4: Advanced Python');
  console.log('═══════════════════════════════════════════');

  await seedModule6(prisma);
  await seedModule7(prisma);

  console.log('');
  console.log('🎉 Stage 4 seeding complete!');
  console.log('  - 2 modules');
  console.log('  - 7 lessons');
  console.log('  - 21 exercises');
  console.log('  - 21 quiz questions');
  console.log('  - 2 projects');
  console.log('  - 12 new skill tags');
}

main()
  .then(async () => {
    await prisma.$disconnect();
  })
  .catch(async (e) => {
    console.error(e);
    await prisma.$disconnect();
    process.exit(1);
  });
