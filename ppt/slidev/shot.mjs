import { chromium } from 'playwright-chromium'
import fs from 'node:fs'

const TOTAL = 26
const out = '/home/zane/session_1/ppt/shots'
fs.mkdirSync(out, { recursive: true })

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } })

// wait for server
for (let i = 0; i < 30; i++) {
  try { await page.goto('http://localhost:3030/1', { timeout: 3000 }); break }
  catch { await new Promise(r => setTimeout(r, 1000)) }
}

for (let n = 1; n <= TOTAL; n++) {
  await page.goto(`http://localhost:3030/${n}?clicks=0`, { waitUntil: 'load' })
  await page.waitForTimeout(1600)
  await page.screenshot({ path: `${out}/slide-${String(n).padStart(2, '0')}.png` })
}
await browser.close()
console.log('done', TOTAL)
