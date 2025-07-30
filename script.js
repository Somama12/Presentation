document.addEventListener('DOMContentLoaded', () => {
    const scrollExplore = document.getElementById('scroll-explore');
    scrollExplore.addEventListener('click', () => {
        document.getElementById('challenge').scrollIntoView({ behavior: 'smooth' });
    });

    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            document.getElementById(targetId).scrollIntoView({ behavior: 'smooth' });
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });

    // Hero Animation
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

    // Wave Animation
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

    // KL Comparison Chart
    const klCanvas = document.getElementById('kl-chart');
    if (klCanvas) {
        const klCtx = klCanvas.getContext('2d');
        const parent = klCanvas.parentElement;
        klCtx.canvas.width = parent.clientWidth * window.devicePixelRatio;
        klCtx.canvas.height = parent.clientHeight * window.devicePixelRatio;
        klCtx.canvas.style.width = `${parent.clientWidth}px`;
        klCtx.canvas.style.height = `${parent.clientHeight}px`;
        klCtx.scale(window.devicePixelRatio, window.devicePixelRatio);

        new Chart(klCtx, {
            type: 'bar',
            data: {
                labels: ['2D', '3D', '4D', '5D'],
                datasets: [
                    {
                        label: 'GMM Gaussian',
                        data: [0.1130, 0.0583, 0.0264, 0.0013],
                        backgroundColor: 'rgba(167, 139, 250, 0.6)',
                        borderColor: 'rgba(167, 139, 250, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'GMM Banana',
                        data: [0.4508, 0.1874, 0.1290, 0.0416],
                        backgroundColor: 'rgba(167, 139, 250, 0.4)',
                        borderColor: 'rgba(167, 139, 250, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'NF Gaussian',
                        data: [0.1381, 0.2798, 0.4146, 0.0771],
                        backgroundColor: 'rgba(94, 234, 212, 0.6)',
                        borderColor: 'rgba(94, 234, 212, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'NF Banana',
                        data: [0.2491, 0.1149, 0.0703, 0.0417],
                        backgroundColor: 'rgba(94, 234, 212, 0.4)',
                        borderColor: 'rgba(94, 234, 212, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#cbd5e1', font: { size: 14 } }
                    },
                    title: {
                        display: true,
                        text: 'Average KL Divergence by Model and Dataset',
                        color: '#cbd5e1',
                        font: { size: 18 }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#94a3b8' },
                        grid: { color: 'rgba(51, 65, 85, 0.5)' },
                        title: { display: true, text: 'Dimension', color: '#cbd5e1' }
                    },
                    y: {
                        ticks: { color: '#94a3b8' },
                        grid: { color: 'rgba(51, 65, 85, 0.5)' },
                        title: { display: true, text: 'Average KL Divergence', color: '#cbd5e1' }
                    }
                }
            }
        });
    }
});