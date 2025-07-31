document.addEventListener('DOMContentLoaded', () => {
    // Existing navigation and scroll logic
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
            const numRipples = 12;
            for (let i = 0; i < numRipples; i++) {
                const phase = time * 1.2 - i * 0.5;
                const radius = maxRadius * (i + 1) / numRipples + Math.sin(phase) * 18;
                ctx.save();
                ctx.beginPath();
                ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
                ctx.strokeStyle = `rgba(129,140,248,${0.35 + 0.15 * Math.sin(phase + i)})`;
                ctx.lineWidth = 5 + 3 * Math.sin(phase + i * 0.7);
                ctx.shadowColor = '#a78bfa';
                ctx.shadowBlur = 32;
                ctx.stroke();
                ctx.restore();
            }
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
    let waveTime = 0;

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
                totalY += wave.amp * Math.sin(x * wave.freq + waveTime + wave.phase);
            });
            totalY += (Math.random() - 0.5) * 15;
            waveCtx.lineTo(x, centerY + totalY);
        }
        waveCtx.strokeStyle = 'rgba(226, 232, 240, 0.8)';
        waveCtx.lineWidth = 2;
        waveCtx.stroke();
        
        waveTime += 0.02;
        requestAnimationFrame(drawWaves);
    }

    resizeWaveCanvas();
    drawWaves();
    window.addEventListener('resize', resizeWaveCanvas);

    // New GMM vs NF Animation (Side-by-Side Comparison)
    const nfGmmCanvas = document.getElementById('nf-gmm-canvas');
    const nfGmmCtx = nfGmmCanvas.getContext('2d');
    let animationRunning = true;
    let nfGmmTime = 0;

    function resizeNfGmmCanvas() {
        const parent = nfGmmCanvas.parentElement;
        nfGmmCanvas.width = parent.clientWidth * window.devicePixelRatio;
        nfGmmCanvas.height = parent.clientHeight * window.devicePixelRatio;
        nfGmmCanvas.style.width = `${parent.clientWidth}px`;
        nfGmmCanvas.style.height = `${parent.clientHeight}px`;
        nfGmmCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }

    function generateBananaPoints(numPoints) {
        const points = [];
        for (let i = 0; i < numPoints; i++) {
            const x = Math.random() * 4 - 2;
            const y = Math.random() * 0.5 + (x * x * 0.5);
            points.push([x, y]);
        }
        return points;
    }

    function drawArrow(ctx, fromX, fromY, toX, toY, color) {
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.stroke();
        // Arrowhead
        const angle = Math.atan2(toY - fromY, toX - fromX);
        ctx.beginPath();
        ctx.moveTo(toX, toY);
        ctx.lineTo(toX - 10 * Math.cos(angle - Math.PI / 6), toY - 10 * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(toX - 10 * Math.cos(angle + Math.PI / 6), toY - 10 * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
    }

    function drawNfGmmAnimation() {
        if (!animationRunning) return;
        nfGmmCtx.clearRect(0, 0, nfGmmCanvas.width, nfGmmCanvas.height);
        const width = nfGmmCanvas.width / window.devicePixelRatio;
        const height = nfGmmCanvas.height / window.devicePixelRatio;
        const halfWidth = width / 2;
        const scaleX = (halfWidth - 40) / 5;
        const scaleY = (height - 40) / 5;
        const offsetXLeft = halfWidth / 2;
        const offsetXRight = halfWidth + halfWidth / 2;
        const offsetY = height / 2;

        // Draw background and divider
        nfGmmCtx.fillStyle = 'rgba(30, 41, 59, 0.8)';
        nfGmmCtx.fillRect(0, 0, width, height);
        nfGmmCtx.strokeStyle = 'rgba(167, 139, 250, 0.6)';
        nfGmmCtx.lineWidth = 2;
        nfGmmCtx.beginPath();
        nfGmmCtx.moveTo(halfWidth, 0);
        nfGmmCtx.lineTo(halfWidth, height);
        nfGmmCtx.stroke();

        // Draw axis labels (Left: GMM)
        nfGmmCtx.fillStyle = '#cbd5e1';
        nfGmmCtx.font = '12px Inter';
        nfGmmCtx.textAlign = 'center';
        nfGmmCtx.fillText('Parameter 1', offsetXLeft, height - 10);
        nfGmmCtx.save();
        nfGmmCtx.translate(10, offsetY);
        nfGmmCtx.rotate(-Math.PI / 2);
        nfGmmCtx.fillText('Parameter 2', 0, 0);
        nfGmmCtx.restore();
        nfGmmCtx.fillText('GMM', offsetXLeft, 20);

        // Draw axis labels (Right: NF)
        nfGmmCtx.fillText('Parameter 1', offsetXRight, height - 10);
        nfGmmCtx.save();
        nfGmmCtx.translate(halfWidth + 10, offsetY);
        nfGmmCtx.rotate(-Math.PI / 2);
        nfGmmCtx.fillText('Parameter 2', 0, 0);
        nfGmmCtx.restore();
        nfGmmCtx.fillText('Normalizing Flow', offsetXRight, 20);

        // Draw axes
        nfGmmCtx.strokeStyle = '#94a3b8';
        nfGmmCtx.lineWidth = 1;
        // Left axes
        nfGmmCtx.beginPath();
        nfGmmCtx.moveTo(offsetXLeft - 2 * scaleX, height - 20);
        nfGmmCtx.lineTo(offsetXLeft + 2 * scaleX, height - 20);
        nfGmmCtx.moveTo(offsetXLeft, 20);
        nfGmmCtx.lineTo(offsetXLeft, height - 20);
        nfGmmCtx.stroke();
        // Right axes
        nfGmmCtx.beginPath();
        nfGmmCtx.moveTo(offsetXRight - 2 * scaleX, height - 20);
        nfGmmCtx.lineTo(offsetXRight + 2 * scaleX, height - 20);
        nfGmmCtx.moveTo(offsetXRight, 20);
        nfGmmCtx.lineTo(offsetXRight, height - 20);
        nfGmmCtx.stroke();

        // Generate and draw Banana distribution points (both sides)
        const bananaPoints = generateBananaPoints(200);
        nfGmmCtx.fillStyle = 'rgba(251, 146, 60, 0.6)';
        bananaPoints.forEach(([x, y]) => {
            // Left (GMM)
            nfGmmCtx.beginPath();
            nfGmmCtx.arc(x * scaleX + offsetXLeft, -y * scaleY + offsetY, 2, 0, 2 * Math.PI);
            nfGmmCtx.fill();
            // Right (NF)
            nfGmmCtx.beginPath();
            nfGmmCtx.arc(x * scaleX + offsetXRight, -y * scaleY + offsetY, 2, 0, 2 * Math.PI);
            nfGmmCtx.fill();
        });

        // Draw GMM contours (Left: three overlapping Gaussian blobs)
        nfGmmCtx.strokeStyle = 'rgba(129, 140, 248, 0.6)';
        nfGmmCtx.lineWidth = 2;
        const gmmCenters = [
            { x: -0.5, y: 0, rx: 0.8, ry: 0.8 },
            { x: 0.5, y: 0, rx: 0.7, ry: 0.9 },
            { x: 0, y: 0.5, rx: 0.9, ry: 0.7 }
        ];
        gmmCenters.forEach((center, i) => {
            const phase = nfGmmTime + i * Math.PI / 2;
            const radiusX = center.rx * scaleX * (1 + 0.1 * Math.sin(phase));
            const radiusY = center.ry * scaleY * (1 + 0.1 * Math.cos(phase));
            nfGmmCtx.beginPath();
            nfGmmCtx.ellipse(
                center.x * scaleX + offsetXLeft,
                -center.y * scaleY + offsetY,
                radiusX,
                radiusY,
                0,
                0,
                2 * Math.PI
            );
            nfGmmCtx.stroke();
        });

        // Draw NF contours (Right: warped banana shape)
        nfGmmCtx.strokeStyle = 'rgba(167, 139, 250, 0.6)';
        nfGmmCtx.beginPath();
        for (let x = -2; x <= 2; x += 0.05) {
            const y = x * x * 0.5 + Math.sin(nfGmmTime) * 0.2;
            nfGmmCtx.lineTo(x * scaleX + offsetXRight, -y * scaleY + offsetY);
        }
        nfGmmCtx.stroke();

        // Draw transformation arrows (Right: to show space deformation)
        const arrowPoints = [
            { from: [-1, 0], to: [-1, -0.5 + Math.sin(nfGmmTime) * 0.2] },
            { from: [0, 0], to: [0, Math.sin(nfGmmTime) * 0.2] },
            { from: [1, 0], to: [1, 0.5 + Math.sin(nfGmmTime) * 0.2] }
        ];
        arrowPoints.forEach(point => {
            drawArrow(
                nfGmmCtx,
                point.from[0] * scaleX + offsetXRight,
                -point.from[1] * scaleY + offsetY,
                point.to[0] * scaleX + offsetXRight,
                -point.to[1] * scaleY + offsetY,
                'rgba(94, 234, 212, 0.8)'
            );
        });

        nfGmmTime += 0.05;
        requestAnimationFrame(drawNfGmmAnimation);
    }

    resizeNfGmmCanvas();
    drawNfGmmAnimation();
    window.addEventListener('resize', resizeNfGmmCanvas);

    // Toggle Animation
    const toggleButton = document.getElementById('toggle-animation');
    toggleButton.addEventListener('click', () => {
        animationRunning = !animationRunning;
        if (animationRunning) {
            drawNfGmmAnimation();
            toggleButton.textContent = 'Pause Animation';
        } else {
            toggleButton.textContent = 'Resume Animation';
        }
    });

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
                        data: [0.0204, 0.0000,  0.0000, 0.0067],
                        backgroundColor: 'rgba(167, 139, 250, 0.6)',
                        borderColor: 'rgba(167, 139, 250, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'GMM Banana',
                        data: [1.4573, 0.6176, 0.4025, 0.3928],
                        backgroundColor: 'rgba(167, 139, 250, 0.4)',
                        borderColor: 'rgba(167, 139, 250, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'NF Gaussian',
                        data: [0.3726, 0.7346, 0.5977, 0.3326],
                        backgroundColor: 'rgba(94, 234, 212, 0.6)',
                        borderColor: 'rgba(94, 234, 212, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'NF Banana',
                        data: [1.1247, 0.5438, 0.2612, 0.0393],
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