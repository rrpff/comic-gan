const { join } = require("path")
const { writeFile } = require("fs").promises
const fetch = require("node-fetch")
const cheerio = require("cheerio")
const mkdirp = require("mkdirp")

const exit = (msg, die) => {
  (die ? console.error : console.log)(msg)
  process.exit(die ? 1 : 0)
}

const [coverGalleryUrl, outputPath] = process.argv.slice(2)
if (!coverGalleryUrl) exit("no cover gallery url given, e.g. https://www.comics.org/series/141/covers/?page=1", true)
if (!outputPath) exit("no relative output path given", true)

const INTERVAL = 1000
const OUTPUT_DIR = join(process.cwd(), outputPath)

const sleep = ms => new Promise((resolve) => setTimeout(resolve, ms))

const scrape = async url => {
  const res = await fetch(url)
  const html = await res.text()

  return cheerio.load(html)
}

const scrapeCovers = async (url, pageNumber = 1, covers = []) => {
  const pageUrl = new URL(url)
  pageUrl.searchParams.set("page", pageNumber)

  const page = await scrape(pageUrl)

  const pageCount = Number(page(".right.pagination li").eq(-3).text())
  const pageCovers = page(".cover_img").map((_, el) => el.attribs.src).get().map(src => src.replace("/w100/", "/w400/"))

  const totalCovers = [...covers, ...pageCovers]

  console.log(`found ${pageCovers.length} on page ${pageNumber}`)

  if (pageCount > pageNumber) {
    await sleep(INTERVAL)
    return await scrapeCovers(url, pageNumber + 1, totalCovers)
  } else {
    return totalCovers
  }
}

;(async () => {
  try {
    const coverUrls = await scrapeCovers(coverGalleryUrl)
    const json = { coverUrls }

    const fpath = join(OUTPUT_DIR, "manifest.json")
    await mkdirp(OUTPUT_DIR)
    await writeFile(fpath, JSON.stringify(json, null, 2))
  } catch (e) {
    console.error(e)
  }
})()
