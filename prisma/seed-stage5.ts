export {};

const { PrismaClient } = require('@prisma/client');
const { seedModule8 } = require('./seed-stage5-module8');

const prisma = new PrismaClient();

async function main() {
  console.log('═══════════════════════════════════════════');
  console.log('  Seeding Stage 5: Deep Learning & Neural Networks');
  console.log('═══════════════════════════════════════════');

  await seedModule8(prisma);

  console.log('');
  console.log('🎉 Stage 5 seeding complete!');
  console.log('  - 1 module');
  console.log('  - 5 lessons');
  console.log('  - 15 exercises');
  console.log('  - 15 quiz questions');
  console.log('  - 2 projects');
  console.log('  - 7 new skill tags');
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
