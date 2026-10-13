// Optional UI check using an existing Playwright installation and local Chrome.
const path = require('node:path');
const fs = require('node:fs');
const { chromium } = require(process.env.VG_PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(__dirname, '..');

(async () => {
  const context = await chromium.launchPersistentContext(path.join(root, '.cache/browser-proof'), {
    executablePath: process.env.VG_CHROME_PATH,
    headless: true,
    viewport: { width: 1440, height: 1080 },
    acceptDownloads: false,
    env: { ...process.env, TEMP: path.join(root, '.cache/tmp'), TMP: path.join(root, '.cache/tmp') },
  });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto('http://127.0.0.1:8501', { waitUntil: 'domcontentloaded' });
    await page.getByRole('button', { name: 'Run synthetic demo' }).click();
    await page.getByText('Demo queued. The video worker will process it.').waitFor();
    let job;
    for (let i = 0; i < 30; i++) {
      const jobs = await (await fetch('http://127.0.0.1:8000/api/jobs')).json();
      job = jobs[0];
      if (job?.status === 'completed') break;
      if (job?.status === 'failed') throw new Error(job.error);
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    if (job?.status !== 'completed') throw new Error('Worker did not complete the demo');
    await page.getByRole('button', { name: 'Refresh job status' }).click();
    await page.getByText('SYNTHETIC DEMO · workflow evidence only;', { exact: false }).waitFor();
    await page.locator('video').waitFor();
    await page.locator('video').evaluate(async video => {
      if (video.readyState < 1) await new Promise((resolve, reject) => {
        video.addEventListener('loadedmetadata', resolve, { once: true });
        video.addEventListener('error', () => reject(new Error('Annotated video cannot play')), { once: true });
        setTimeout(() => reject(new Error('Video metadata timeout')), 15000);
      });
      if (!(video.duration > 0)) throw new Error('Video has no duration');
    });
    await page.getByRole('button', { name: 'Download evidence JSON' }).waitFor();
    await page.getByText('Frames inspected', { exact: true }).waitFor();
    await page.waitForTimeout(1000);
    await page.screenshot({ path: path.join(root, 'reports/dashboard-analysis.png'), fullPage: true });
    await page.getByText('Event log', { exact: true }).click();
    const panel = page.locator('[data-testid="stExpander"]').filter({ hasText: job.id.slice(0, 8) });
    await panel.locator('summary').click();
    await panel.getByRole('textbox', { name: 'Review note' }).fill('Synthetic browser verification');
    await panel.getByRole('button', { name: 'Save review' }).click();
    let events = [];
    for (let i = 0; i < 20; i++) {
      events = await (await fetch(`http://127.0.0.1:8000/api/events?job_id=${job.id}`)).json();
      if (events.some(event => event.note === 'Synthetic browser verification')) break;
      await new Promise(resolve => setTimeout(resolve, 250));
    }
    if (!events.some(event => event.note === 'Synthetic browser verification')) {
      throw new Error('Review note was not persisted');
    }
    await page.screenshot({ path: path.join(root, 'reports/dashboard-review.png'), fullPage: true });
    await page.getByText('Analytics', { exact: true }).click();
    await page.getByText('Observed person-minutes', { exact: true }).waitFor();
    await page.screenshot({ path: path.join(root, 'reports/dashboard-analytics.png'), fullPage: true });
    await page.getByText('Model & data', { exact: true }).click();
    await page.getByText('Model and data status', { exact: true }).waitFor();
    const model = await (await fetch('http://127.0.0.1:8000/api/model')).json();
    let realUpload = false;
    if (model.status === 'experimental') {
      await page.getByText('At the evaluated confidence threshold', { exact: true }).waitFor();
      const splitLabel = model.report.evaluation.split === 'val' ? 'Validation images' : 'Test images';
      await page.getByText(splitLabel, { exact: true }).waitFor();
      if (model.report.latest_experiment) {
        await page.getByText('Latest training experiment', { exact: true }).waitFor();
        await page.getByText(model.report.latest_experiment.summary, { exact: true }).last().waitFor();
      }
      await page.getByText('Known limits', { exact: true }).waitFor();
      await page.getByRole('button', { name: 'Stop', exact: true }).waitFor({ state: 'hidden' });
    }
    await page.screenshot({ path: path.join(root, 'reports/dashboard-model.png'), fullPage: true });
    if (model.status === 'experimental') {
      await page.getByText('Analyze & review', { exact: true }).click();
      await page.getByText('Analyze your own image or recorded video', { exact: true }).click();
      await page.locator('input[type="file"]').setInputFiles(path.join(root, model.report.sample));
      await page.getByRole('button', { name: 'Analyze upload', exact: true }).click();
      await page.getByText('EXPERIMENTAL MODEL · predictions need review', { exact: false }).waitFor({timeout: 60000});
      await page.locator('[data-testid="stImage"] img').waitFor();
      await page.getByRole('button', { name: 'Download evidence JSON' }).waitFor();
      await page.getByRole('button', { name: 'Stop', exact: true }).waitFor({ state: 'hidden' });
      await page.screenshot({ path: path.join(root, 'reports/dashboard-real-image.png'), fullPage: true });
      realUpload = true;
    }
    if (await page.locator('[data-testid="stException"]').count()) throw new Error('Streamlit exception');
    if (errors.length) throw new Error(errors.join('\n'));
    const report = { status: 'passed', job: job.id, pages: 4, video_metadata: 'loaded',
      review_persisted: true, real_image_upload: realUpload, browser_errors: errors,
      model_evaluation_split: model.report?.evaluation?.split,
      candidate_comparison_visible: Boolean(model.report?.latest_experiment),
      scope: 'Application workflow and image inference, not new accuracy evidence' };
    fs.writeFileSync(path.join(root, 'reports/browser-check.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
  } catch (error) {
    await page.screenshot({ path: path.join(root, 'reports/browser-failure.png'), fullPage: true });
    throw error;
  } finally {
    await context.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
