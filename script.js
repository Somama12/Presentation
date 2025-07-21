document.addEventListener('DOMContentLoaded', () => {

    // --- Gravitational Wave Ripple Effect for Hero ---
    const gwCanvas = document.getElementById('gw-ripple-canvas');
    if (gwCanvas) {
        const ctx = gwCanvas.getContext('2d');
        let width = window.innerWidth;
        let height = window.innerHeight;
        let time = 0;

        function resizeGWCanvas() {
            width = window.innerWidth;
            height = window.innerHeight;
            gwCanvas.width = width * window.devicePixelRatio;
            gwCanvas.height = height * window.devicePixelRatio;
            gwCanvas.style.width = `${width}px`;
            gwCanvas.style.height = `${height}px`;
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        }

        function drawGWRipples() {
            ctx.clearRect(0, 0, width, height);
            const centerX = width / 2;
            const centerY = height / 2;
            const maxRadius = Math.min(width, height) * 0.45;
            const numRipples = 12; // Increased for more ripples
            for (let i = 0; i < numRipples; i++) {
                const phase = time * 1.2 - i * 0.5;
                const radius = maxRadius * (i + 1) / numRipples + Math.sin(phase) * 18;
                ctx.save();
                ctx.beginPath();
                ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
                // More vibrant and visible color
                ctx.strokeStyle = `rgba(129,140,248,${0.35 + 0.15 * Math.sin(phase + i)})`;
                ctx.lineWidth = 5 + 3 * Math.sin(phase + i * 0.7); // Thicker lines
                ctx.shadowColor = '#a78bfa';
                ctx.shadowBlur = 32; // More blur
                ctx.stroke();
                ctx.restore();
            }
            // Add a more visible "black hole" center
            ctx.save();
            ctx.beginPath();
            ctx.arc(centerX, centerY, maxRadius * 0.13 + Math.sin(time) * 2, 0, 2 * Math.PI);
            ctx.fillStyle = '#0f172a';
            ctx.shadowColor = '#818cf8';
            ctx.shadowBlur = 48;
            ctx.fill();
            ctx.restore();
            time += 0.012;
            requestAnimationFrame(drawGWRipples);
        }
        resizeGWCanvas();
        drawGWRipples();
        window.addEventListener('resize', resizeGWCanvas);
    }

    // --- Wave Animation ---
    const waveCanvas = document.getElementById('wave-canvas');
    const waveCtx = waveCanvas.getContext('2d');
    let time = 0;
    
    const waves = [
        { amp: 50, freq: 0.02, phase: 0, color: 'rgba(167, 139, 250, 0.6)' },
        { amp: 30, freq: 0.05, phase: 1.5, color: 'rgba(129, 140, 248, 0.6)' },
        { amp: 20, freq: 0.08, phase: 3, color: 'rgba(94, 234, 212, 0.6)' },
        { amp: 15, freq: 0.12, phase: 4.5, color: 'rgba(251, 146, 60, 0.5)' }
    ];

    function resizeWaveCanvas() {
        const parent = waveCanvas.parentElement;
        waveCanvas.width = parent.clientWidth * window.devicePixelRatio;
        waveCanvas.height = parent.clientHeight * window.devicePixelRatio;
        waveCanvas.style.width = `${parent.clientWidth}px`;
        waveCanvas.style.height = `${parent.clientHeight}px`;
        waveCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }

    function drawWaves() {
        waveCtx.clearRect(0, 0, waveCanvas.width, waveCanvas.height);
        const centerY = waveCanvas.height / window.devicePixelRatio / 2;

        waveCtx.beginPath();
        for (let x = 0; x < waveCanvas.width / window.devicePixelRatio; x++) {
            let totalY = 0;
            waves.forEach(wave => {
                totalY += wave.amp * Math.sin(x * wave.freq + time + wave.phase);
            });
            totalY += (Math.random() - 0.5) * 15;
            waveCtx.lineTo(x, centerY + totalY);
        }
        waveCtx.strokeStyle = 'rgba(226, 232, 240, 0.8)';
        waveCtx.lineWidth = 2;
        waveCtx.stroke();
        
        time += 0.02;
        requestAnimationFrame(drawWaves);
    }

    resizeWaveCanvas();
    drawWaves();
    window.addEventListener('resize', resizeWaveCanvas);

    // --- Navigation ---
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('main section');
    const scrollExploreBtn = document.getElementById('scroll-explore');

    scrollExploreBtn.addEventListener('click', () => {
        document.querySelector('#challenge').scrollIntoView({ behavior: 'smooth' });
    });

    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            if (pageYOffset >= sectionTop - 68) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href').substring(1) === current) {
                link.classList.add('active');
            }
        });
    });
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href');
            document.querySelector(targetId).scrollIntoView({ behavior: 'smooth' });
        });
    });

    // --- Interactive Dashboard Logic ---
    const gmmTabBtn = document.getElementById('gmm-tab-btn');
    const nfTabBtn = document.getElementById('nf-tab-btn');
    const gmmContent = document.getElementById('gmm-content');
    const nfContent = document.getElementById('nf-content');
    
    const loadGaussianBtn = document.getElementById('load-gaussian-btn');
    const loadBananaBtn = document.getElementById('load-banana-btn');
    const fitGmmBtn = document.getElementById('fit-gmm-btn');
    const gmmResultsEl = document.getElementById('gmm-results');
    const klScoreEl = document.getElementById('kl-score');
    const conclusionEl = document.getElementById('conclusion-text');
    const datasetControls = document.querySelectorAll('.dataset-controls');
    
    let gmmChart, nfChart;
    let currentDataset = null;

    const datasets = {
        gaussian: {
            name: 'Gaussian Mix',
            kl: '0.2109',
            conclusion: 'Good Fit! GMMs are well-suited for this type of simple, Gaussian-like distribution.',
            generator: () => {
                const data = [];
                for(let i = 0; i < 300; i++) data.push({x: normalRandom(0, 1), y: normalRandom(0, 1)});
                for(let i = 0; i < 300; i++) data.push({x: normalRandom(5, 0.5), y: normalRandom(5, 0.5)});
                return data;
            },
            gmmSamples: () => {
                const data = [];
                for(let i = 0; i < 600; i++) {
                     if (Math.random() < 0.5) {
                        data.push({x: normalRandom(0, 1), y: normalRandom(0, 1)});
                     } else {
                        data.push({x: normalRandom(5, 0.5), y: normalRandom(5, 0.5)});
                     }
                }
                return data;
            }
        },
        banana: {
            name: 'Banana',
            kl: '0.4592',
            conclusion: 'Poor Fit. The GMM fails to capture the non-linear "banana" shape, revealing its limitations.',
            generator: () => {
                const data = [];
                for(let i = 0; i < 300; i++) {
                    const x = normalRandom(0, 1);
                    const y = x*x + normalRandom(0, 0.5);
                    data.push({x: x, y: y});
                }
                return data;
            },
            gmmSamples: () => {
                const data = [];
                for(let i = 0; i < 300; i++) {
                     if (Math.random() < 0.5) {
                        data.push({x: normalRandom(-0.8, 0.7), y: normalRandom(0.8, 1.2)});
                     } else {
                        data.push({x: normalRandom(0.8, 0.7), y: normalRandom(0.8, 1.2)});
                     }
                }
                return data;
            }
        }
    };
    
    function normalRandom(mean = 0, stdDev = 1) {
        let u1 = Math.random();
        let u2 = Math.random();
        let randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
        return mean + stdDev * randStdNormal;
    }

    function createGmmChart(data = []) {
        if (gmmChart) gmmChart.destroy();
        const ctx = document.getElementById('gmm-chart').getContext('2d');
        const parent = ctx.canvas.parentElement;
        ctx.canvas.width = parent.clientWidth * window.devicePixelRatio;
        ctx.canvas.height = parent.clientHeight * window.devicePixelRatio;
        ctx.canvas.style.width = `${parent.clientWidth}px`;
        ctx.canvas.style.height = `${parent.clientHeight}px`;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        gmmChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Original Data',
                    data: data,
                    backgroundColor: 'rgba(167, 139, 250, 0.6)',
                    pointRadius: 3,
                    pointHoverRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#cbd5e1' } } },
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(51, 65, 85, 0.5)' } },
                    y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(51, 65, 85, 0.5)' } }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeOutQuad'
                }
            }
        });
    }

    // --- Original createNfChart function ---
    function createNfChart() {
        if (nfChart) nfChart.destroy();
        const ctx = document.getElementById('nf-chart').getContext('2d');
        const parent = ctx.canvas.parentElement;
        ctx.canvas.width = parent.clientWidth * window.devicePixelRatio;
        ctx.canvas.height = parent.clientHeight * window.devicePixelRatio;
        ctx.canvas.style.width = `${parent.clientWidth}px`;
        ctx.canvas.style.height = `${parent.clientHeight}px`;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        const realData = datasets.gaussian.generator();
        const gmmData = datasets.gaussian.gmmSamples();
        const flowData = (() => {
            const data = [];
            for(let i = 0; i < 600; i++) {
                if (Math.random() < 0.5) {
                    data.push({x: normalRandom(-0.5, 1.2), y: normalRandom(-0.5, 1.2)});
                } else {
                    data.push({x: normalRandom(4.5, 0.8), y: normalRandom(4.5, 0.8)});
                }
            }
            return data;
        })();
        nfChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    { label: 'Real Data', data: realData, backgroundColor: 'rgba(255, 255, 255, 0.4)', pointRadius: 2 },
                    { label: 'GMM Samples', data: gmmData, backgroundColor: 'rgba(167, 139, 250, 0.7)', pointRadius: 3 },
                    { label: 'Flow Samples (WIP)', data: flowData, backgroundColor: 'rgba(94, 234, 212, 0.7)', pointRadius: 3 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#cbd5e1' } } },
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(51, 65, 85, 0.5)' } },
                    y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(51, 65, 85, 0.5)' } }
                },
                animation: { duration: 1000, easing: 'easeOutQuad' }
            }
        });
    }

    gmmTabBtn.addEventListener('click', () => {
        gmmContent.classList.remove('hidden');
        nfContent.classList.add('hidden');
        gmmTabBtn.classList.add('active');
        nfTabBtn.classList.remove('active');
        datasetControls.forEach(control => control.classList.remove('hidden'));
        createGmmChart(currentDataset ? currentDataset.generator() : []);
    });

    // --- Restore NF tab event listener ---
    nfTabBtn.addEventListener('click', () => {
        nfContent.classList.remove('hidden');
        gmmContent.classList.add('hidden');
        nfTabBtn.classList.add('active');
        gmmTabBtn.classList.remove('active');
        datasetControls.forEach(control => control.classList.add('hidden'));
        createNfChart();
    });
    
    function updateDatasetSelection(btn) {
        [loadGaussianBtn, loadBananaBtn].forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        fitGmmBtn.disabled = false;
        gmmResultsEl.classList.add('hidden');
    }

    loadGaussianBtn.addEventListener('click', () => {
        currentDataset = datasets.gaussian;
        updateDatasetSelection(loadGaussianBtn);
        if (gmmContent.classList.contains('hidden')) {
            createNfChart(); // Ensure NF chart updates if active
        } else {
            createGmmChart(currentDataset.generator());
        }
    });
    
    loadBananaBtn.addEventListener('click', () => {
        currentDataset = datasets.banana;
        updateDatasetSelection(loadBananaBtn);
        if (gmmContent.classList.contains('hidden')) {
            createNfChart(); // Ensure NF chart updates if active
        } else {
            createGmmChart(currentDataset.generator());
        }
    });

    fitGmmBtn.addEventListener('click', () => {
        if (!currentDataset) return;

        const gmmSamples = currentDataset.gmmSamples();
        gmmChart.data.datasets.push({
            label: 'GMM Samples',
            data: gmmSamples,
            backgroundColor: 'rgba(251, 146, 60, 0.6)',
            pointRadius: 3
        });
        gmmChart.update();

        klScoreEl.textContent = currentDataset.kl;
        conclusionEl.textContent = currentDataset.conclusion;
        gmmResultsEl.classList.remove('hidden');
        fitGmmBtn.disabled = true;
    });

    // --- NF Tab Controls ---
    const loadGaussianBtnNF = document.getElementById('load-gaussian-btn-nf');
    const loadBananaBtnNF = document.getElementById('load-banana-btn-nf');
    const fitNfBtn = document.getElementById('fit-nf-btn');
    let nfSelectedDataset = null;
    let nfRealShown = false;
    let nfFitPerformed = false;
    function createNfChartStep({ showReal = false, showFit = false } = {}) {
        if (nfChart) nfChart.destroy();
        const ctx = document.getElementById('nf-chart').getContext('2d');
        const parent = ctx.canvas.parentElement;
        ctx.canvas.width = parent.clientWidth * window.devicePixelRatio;
        ctx.canvas.height = parent.clientHeight * window.devicePixelRatio;
        ctx.canvas.style.width = `${parent.clientWidth}px`;
        ctx.canvas.style.height = `${parent.clientHeight}px`;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        let realData = [];
        let gmmData = [];
        let flowData = [];
        if (nfSelectedDataset === 'gaussian') {
            realData = datasets.gaussian.generator();
            gmmData = datasets.gaussian.gmmSamples();
            flowData = (() => {
                const data = [];
                for(let i = 0; i < 600; i++) {
                    if (Math.random() < 0.5) {
                        data.push({x: normalRandom(-0.5, 1.2), y: normalRandom(-0.5, 1.2)});
                    } else {
                        data.push({x: normalRandom(4.5, 0.8), y: normalRandom(4.5, 0.8)});
                    }
                }
                return data;
            })();
        } else if (nfSelectedDataset === 'banana') {
            realData = datasets.banana.generator();
            gmmData = datasets.banana.gmmSamples();
            flowData = (() => {
                const data = [];
                for(let i = 0; i < 300; i++) {
                    const x = normalRandom(0, 1);
                    const y = x*x + normalRandom(0, 0.5);
                    data.push({x, y});
                }
                return data;
            })();
        }
        const datasetsToShow = [];
        if (showReal) {
            datasetsToShow.push({ label: 'Real Data', data: realData, backgroundColor: 'rgba(255, 255, 255, 0.4)', pointRadius: 2 });
        }
        if (showFit) {
            datasetsToShow.push(
                { label: 'GMM Samples', data: gmmData, backgroundColor: 'rgba(167, 139, 250, 0.7)', pointRadius: 3 },
                { label: 'Flow Samples (WIP)', data: flowData, backgroundColor: 'rgba(94, 234, 212, 0.7)', pointRadius: 3 }
            );
        }
        nfChart = new Chart(ctx, {
            type: 'scatter',
            data: { datasets: datasetsToShow },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#cbd5e1' } } },
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(51, 65, 85, 0.5)' } },
                    y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(51, 65, 85, 0.5)' } }
                },
                animation: { duration: 1000, easing: 'easeOutQuad' }
            }
        });
    }
    function resetNfStepControls() {
        nfSelectedDataset = null;
        nfRealShown = false;
        nfFitPerformed = false;
        if (nfChart) nfChart.destroy();
        [loadGaussianBtnNF, loadBananaBtnNF].forEach(b => b && b.classList.remove('active'));
        showRealNfBtn.disabled = true;
        fitNfBtn.disabled = true;
        nfResultsEl.classList.add('hidden');
        klScoreNfEl.textContent = '';
        conclusionNfEl.textContent = '';
    }
    if (loadGaussianBtnNF && loadBananaBtnNF && showRealNfBtn && fitNfBtn) {
        loadGaussianBtnNF.addEventListener('click', () => {
            nfSelectedDataset = 'gaussian';
            nfRealShown = false;
            nfFitPerformed = false;
            showRealNfBtn.disabled = false;
            fitNfBtn.disabled = true;
            createNfChartStep({ showReal: false, showFit: false });
            loadGaussianBtnNF.classList.add('active');
            loadBananaBtnNF.classList.remove('active');
            nfResultsEl.classList.add('hidden');
        });
        loadBananaBtnNF.addEventListener('click', () => {
            nfSelectedDataset = 'banana';
            nfRealShown = false;
            nfFitPerformed = false;
            showRealNfBtn.disabled = false;
            fitNfBtn.disabled = true;
            createNfChartStep({ showReal: false, showFit: false });
            loadBananaBtnNF.classList.add('active');
            loadGaussianBtnNF.classList.remove('active');
            nfResultsEl.classList.add('hidden');
        });
        showRealNfBtn.addEventListener('click', () => {
            if (!nfSelectedDataset) return;
            nfRealShown = true;
            nfFitPerformed = false;
            createNfChartStep({ showReal: true, showFit: false });
            showRealNfBtn.disabled = true;
            fitNfBtn.disabled = false;
            nfResultsEl.classList.add('hidden');
        });
        fitNfBtn.addEventListener('click', () => {
            if (!nfSelectedDataset || !nfRealShown) return;
            nfFitPerformed = true;
            createNfChartStep({ showReal: true, showFit: true });
            fitNfBtn.disabled = true;
            klScoreNfEl.textContent = nfDatasetKLs[nfSelectedDataset].kl;
            conclusionNfEl.textContent = nfDatasetKLs[nfSelectedDataset].conclusion;
            nfResultsEl.classList.remove('hidden');
        });
    }
    nfTabBtn.addEventListener('click', () => {
        resetNfStepControls();
    });
    gmmTabBtn.addEventListener('click', () => {
        resetNfStepControls();
    });

    // --- NF Results Logic ---
    const nfResultsEl = document.getElementById('nf-results');
    const klScoreNfEl = document.getElementById('kl-score-nf');
    const conclusionNfEl = document.getElementById('conclusion-text-nf');
    const klGaussianNfEl = document.getElementById('kl-gaussian-nf');
    const klBananaNfEl = document.getElementById('kl-banana-nf');
    const nfDatasetKLs = {
        gaussian: { kl: '0.2109', conclusion: 'Good Fit! NFs are well-suited for this type of simple, Gaussian-like distribution.' },
        banana: { kl: '0.4592', conclusion: 'Excellent Fit! NFs capture the non-linear "banana" shape, outperforming GMMs.' }
    };
    function showNfResults(dataset) {
        if (!dataset) return;
        klScoreNfEl.textContent = nfDatasetKLs[dataset].kl;
        conclusionNfEl.textContent = nfDatasetKLs[dataset].conclusion;
        nfResultsEl.classList.remove('hidden');
    }
    function hideNfResults() {
        nfResultsEl.classList.add('hidden');
        klScoreNfEl.textContent = '';
        conclusionNfEl.textContent = '';
    }
    // Update NF controls to show/hide results
    if (loadGaussianBtnNF && loadBananaBtnNF && fitNfBtn) {
        loadGaussianBtnNF.addEventListener('click', () => {
            hideNfResults();
        });
        loadBananaBtnNF.addEventListener('click', () => {
            hideNfResults();
        });
        fitNfBtn.addEventListener('click', () => {
            if (!nfSelectedDataset) return;
            showNfResults(nfSelectedDataset);
        });
    }
    nfTabBtn.addEventListener('click', () => {
        hideNfResults();
    });
    gmmTabBtn.addEventListener('click', () => {
        hideNfResults();
    });

    // --- Code Display Modal Logic ---
    const codeModal = document.getElementById('code-modal');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const modalTitle = document.getElementById('modal-title');
    const codeDisplay = document.getElementById('code-display');
    const showGmmCodeBtn = document.getElementById('show-gmm-code-btn');
    const showNfCodeBtn = document.getElementById('show-nf-code-btn');

    const gmmCode = `
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy

np.random.seed(42)

# Gaussian Clusters
gaussian1 = np.random.normal(loc=0, scale=1.0, size=(300, 2))
gaussian2 = np.random.normal(loc=5, scale=0.5, size=(300, 2))
X = np.vstack([gaussian1, gaussian2])

# Non-Gaussian: Banana shape
x = np.random.normal(0, 1, 300)
y = x**2 + np.random.normal(0, 0.1, 300)
banana = np.stack([x, y], axis=1)

# Fit GMMs and Generate Samples
# GMM on Gaussian data
gmm = GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(X)
generated_gaussian, _ = gmm.sample(n_samples=600)

# GMM on Banana data
gmm_banana = GaussianMixture(n_components=2, covariance_type='full')
gmm_banana.fit(banana)
generated_banana, _ = gmm_banana.sample(n_samples=300)

# KL Divergence Function
def compute_kl_divergence(true_data, generated_data, bins=50):
    hist_true, bin_edges = np.histogram(true_data, bins=bins, density=True)
    hist_gen, _ = np.histogram(generated_data, bins=bin_edges, density=True)
    hist_true += 1e-8
    hist_gen += 1e-8
    hist_true /= np.sum(hist_true)
    hist_gen /= np.sum(hist_gen)
    return entropy(hist_true, hist_gen)

# KL Divergence Scores
kl_x = compute_kl_divergence(X[:, 0], generated_gaussian[:, 0])
kl_y = compute_kl_divergence(X[:, 1], generated_gaussian[:, 1])

kl_bx = compute_kl_divergence(banana[:, 0], generated_banana[:, 0])
kl_by = compute_kl_divergence(banana[:, 1], generated_banana[:, 1])
`;

    const nfCode = `
# !pip install nflows==0.14
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from nflows.flows.base import Flow
from nflows.transforms import CompositeTransform, ReversePermutation, PiecewiseRationalQuadraticCouplingTransform
from nflows.nn.nets import ResidualNet
from scipy.stats import gaussian_kde

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 1. Generate 2D Gaussian Mixture Data
mean1 = [0, 0]
mean2 = [4, 4]
cov = [[1, 0.5], [0.5, 1]]
gauss1_2d = np.random.multivariate_normal(mean1, cov, 500)
gauss2_2d = np.random.multivariate_normal(mean2, cov, 500)
data_2d = np.vstack([gauss1_2d, gauss2_2d])

# Normalize Data
scaler = StandardScaler()
data_2d_scaled = scaler.fit_transform(data_2d)

# 2. Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(data_2d_scaled)
gmm_log_likelihood = gmm.score(data_2d_scaled) * len(data_2d_scaled)

# 3. Custom Mixture of Gaussians for Flow Base Distribution
class CustomMixtureOfGaussians(torch.distributions.Distribution):
    arg_constraints = {}
    def __init__(self, means, variances, weights, input_dim=2, device='cpu'):
        super().__init__(validate_args=False)
        self.input_dim = input_dim
        self.distributions = [
            torch.distributions.MultivariateNormal(
                torch.tensor(m, dtype=torch.float32, device=device),
                torch.diag(torch.tensor(v, dtype=torch.float32, device=device))
            ) for m, v in zip(means, variances)
        ]
        self.weights = torch.tensor(weights, dtype=torch.float32, device=device)

    def log_prob(self, x, context=None):
        log_probs = torch.stack([d.log_prob(x) + torch.log(w) for d, w in zip(self.distributions, self.weights)], dim=0)
        return torch.logsumexp(log_probs, dim=0)

    def sample(self, sample_shape, context=None):
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        z = torch.multinomial(self.weights, sample_shape[0], replacement=True)
        samples = torch.stack([self.distributions[i].sample(sample_shape) for i in range(len(self.distributions))])
        return samples[z, torch.arange(sample_shape[0])]

# 4. Create and Train Normalizing Flow
def create_flow(input_dim=2, hidden_features=128, num_layers=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = []
    masks = [torch.tensor([1, 0], dtype=torch.float32, device=device),
             torch.tensor([0, 1], dtype=torch.float32, device=device)] * (num_layers // 2)
    for mask in masks:
        transforms.append(ReversePermutation(features=input_dim))
        transforms.append(
            PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_features,
                    num_blocks=2,
                    activation=nn.ReLU()
                ),
                tails='linear',
                num_bins=4
            )
        )
    base_dist = CustomMixtureOfGaussians(
        means=[[-1.0, -1.0], [1.0, 1.0]],
        variances=[[1.0, 1.0], [1.0, 1.0]],
        weights=[0.5, 0.5],
        input_dim=input_dim,
        device=device
    )
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist).to(device)
    return flow

def train_flow(flow, data, steps=5000, lr=5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow = flow.to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    losses = []
    batch_size = 64
    best_loss = float('inf')
    patience = 500
    steps_without_improvement = 0
    for step in range(steps):
        idx = np.random.choice(len(data), size=batch_size, replace=false)
        batch = data_tensor[idx]
        try:
            log_prob = flow.log_prob(batch)
            loss = -log_prob.mean()
            if not torch.isfinite(loss):
                break
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            if loss.item() < best_loss:
                best_loss = loss.item()
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                if steps_without_improvement >= patience:
                    break
        except Exception as e:
            raise
    flow_log_likelihood = flow.log_prob(data_tensor).sum().item()
    return flow, losses, flow_log_likelihood

# 5. Evaluate Models
def compute_kl_divergence(real_data, generated_data):
    kl_scores = []
    for i in range(2):
        kde_real = gaussian_kde(real_data[:, i])
        kde_gen = gaussian_kde(generated_data[:, i])
        xs = np.linspace(
            min(real_data[:, i].min(), generated_data[:, i].min()),
            max(real_data[:, i].max(), generated_data[:, i].max()), 1000
        )
        p = np.clip(kde_real(xs), 1e-10, 1)
        q = np.clip(kde_gen(xs), 1e-10, 1)
        kl = np.sum(p * np.log(p / q))
        kl_scores.append(kl)
    return np.mean(kl_scores)
`;

    function showCodeModal(title, code) {
        modalTitle.textContent = title;
        codeDisplay.textContent = code;
        codeModal.classList.add('show');
    }

    modalCloseBtn.addEventListener('click', () => {
        codeModal.classList.remove('show');
    });

    showGmmCodeBtn.addEventListener('click', () => {
        showCodeModal('GMM Model Code', gmmCode);
    });

    showNfCodeBtn.addEventListener('click', () => {
        showCodeModal('Normalizing Flow Model Code', nfCode);
    });

    // --- Model Chart Visualizations ---
    let gmmModelChart, nfModelChart;
    // GMM Model Chart: Gaussian Mixture
    const gmmModelCanvas = document.getElementById('gmm-chart-model');
    if (gmmModelCanvas) {
        const ctx = gmmModelCanvas.getContext('2d');
        const parent = gmmModelCanvas.parentElement;
        gmmModelCanvas.width = parent.clientWidth * window.devicePixelRatio;
        gmmModelCanvas.height = parent.clientHeight * window.devicePixelRatio;
        gmmModelCanvas.style.width = `${parent.clientWidth}px`;
        gmmModelCanvas.style.height = `${parent.clientHeight}px`;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        // Gaussian Mixture Data
        const gmmData = [];
        for(let i = 0; i < 300; i++) gmmData.push({x: normalRandom(0, 1), y: normalRandom(0, 1)});
        for(let i = 0; i < 300; i++) gmmData.push({x: normalRandom(5, 0.5), y: normalRandom(5, 0.5)});
        gmmModelChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Gaussian Mixture',
                    data: gmmData,
                    backgroundColor: 'rgba(167, 139, 250, 0.6)',
                    pointRadius: 3,
                    pointHoverRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#cbd5e1' } } },
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(51, 65, 85, 0.5)' } },
                    y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(51, 65, 85, 0.5)' } }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeOutQuad'
                }
            }
        });
    }
    // NF Model Chart: Single Normal
    const nfModelCanvas = document.getElementById('nf-chart-model');
    if (nfModelCanvas) {
        const ctx = nfModelCanvas.getContext('2d');
        const parent = nfModelCanvas.parentElement;
        nfModelCanvas.width = parent.clientWidth * window.devicePixelRatio;
        nfModelCanvas.height = parent.clientHeight * window.devicePixelRatio;
        nfModelCanvas.style.width = `${parent.clientWidth}px`;
        nfModelCanvas.style.height = `${parent.clientHeight}px`;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        // Single Normal Data
        const nfData = [];
        for(let i = 0; i < 600; i++) nfData.push({x: normalRandom(0, 1), y: normalRandom(0, 1)});
        nfModelChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Normal Distribution',
                    data: nfData,
                    backgroundColor: 'rgba(94, 234, 212, 0.7)',
                    pointRadius: 3,
                    pointHoverRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#cbd5e1' } } },
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(51, 65, 85, 0.5)' } },
                    y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(51, 65, 85, 0.5)' } }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeOutQuad'
                }
            }
        });
    }

    // Initial state
    createGmmChart();
});