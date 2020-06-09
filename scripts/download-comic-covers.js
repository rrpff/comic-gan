const { join, basename, dirname } = require("path")
const fetch = require("node-fetch")
const sharp = require("sharp")

const exit = (msg, die) => {
  (die ? console.error : console.log)(msg)
  process.exit(die ? 1 : 0)
}

const [inputPath] = process.argv.slice(2)
if (!inputPath) exit("no input manifest.json given", true)

const INTERVAL = 1000
const MANIFEST_PATH = join(process.cwd(), inputPath)
const OUTPUT_PATH = dirname(MANIFEST_PATH)

const sleep = ms => new Promise((resolve) => setTimeout(resolve, ms))

const download = async (coverUrls, count) => {
  if (coverUrls.length === 0) return

  const url = coverUrls[0]
  const imageName = basename(new URL(url).pathname)
  const res = await fetch(url)
  const buf = await res.buffer()
  const fname = join(OUTPUT_PATH, imageName)

  await sharp(buf)
    .resize({ width: 512, height: 512, fit: "contain" })
    .jpeg({ quality: 100 })
    .toFile(fname)

  console.log(`saved. ${coverUrls.length - 1} remaining...`)

  await sleep(INTERVAL)
  await download(coverUrls.slice(1))
}

;(async () => {
  try {
    const { coverUrls } = require(MANIFEST_PATH)
    await download(coverUrls)
  } catch (e) {
    console.error(e)
  }
})()
