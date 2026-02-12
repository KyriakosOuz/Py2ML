export {};

const { PrismaClient } = require('@prisma/client');
const bcrypt = require('bcryptjs');

const prisma = new PrismaClient();

async function main() {
  const email = 'kyriakos.ouzounis@gmail.com';
  const name = 'Kyriakos Ouzounis';
  const password = process.env.ADMIN_PASSWORD || 'Py2ML-Admin-2024!';

  const passwordHash = await bcrypt.hash(password, 12);

  const user = await prisma.user.upsert({
    where: { email },
    update: { role: 'ADMIN', name, passwordHash },
    create: {
      email,
      name,
      passwordHash,
      role: 'ADMIN',
    },
  });

  console.log(`Admin user ready: ${user.email} (${user.role})`);
  console.log(`Password: ${password}`);
  console.log('Change your password after first login if using the default.');
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(() => prisma.$disconnect());
