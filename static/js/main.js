// Client-side interactions and charts for Rising Waters Flood Prediction System

// Wait for DOM to load
document.addEventListener("DOMContentLoaded", () => {
    // 1. Initialize gauge if container exists
    initGauge();
    
    // 2. Initialize Model Evaluation charts if on models page
    if (document.getElementById("accuracyChart")) {
        initModelCharts();
    }
    
    // 3. Initialize Control Room District Simulator
    if (document.getElementById("districtContainer")) {
        initControlRoom();
    }
    
    // 4. Form validation and range input displays
    initRangeDisplay();
});

// Dynamic Gauge Widget updates
function initGauge() {
    const gaugeValueEl = document.getElementById("gaugeValue");
    if (!gaugeValueEl) return;
    
    const confidence = parseFloat(gaugeValueEl.dataset.confidence || 0);
    const prediction = parseInt(gaugeValueEl.dataset.prediction || 0);
    
    const circle = document.querySelector(".gauge-circle-val");
    if (circle) {
        const radius = circle.r.baseVal.value;
        const circumference = 2 * Math.PI * radius;
        
        // Stroke dasharray represents [length_of_dash, length_of_gap]
        circle.style.strokeDasharray = `${circumference} ${circumference}`;
        
        // Calculate offset (e.g. 80% means 20% gap)
        const offset = circumference - (confidence / 100) * circumference;
        
        // Set stroke color based on prediction
        if (prediction === 1) {
            circle.style.stroke = "#ef4444"; // Red for High Risk
        } else {
            circle.style.stroke = "#10b981"; // Green for Low Risk
        }
        
        // Trigger animation
        setTimeout(() => {
            circle.style.strokeDashoffset = offset;
        }, 100);
    }
}

// Display selected range values dynamically in the forms
function initRangeDisplay() {
    const rangeInputs = document.querySelectorAll(".range-input");
    rangeInputs.forEach(input => {
        const displayId = `${input.id}_val`;
        const displayEl = document.getElementById(displayId);
        
        if (displayEl) {
            input.addEventListener("input", () => {
                displayEl.textContent = input.value;
            });
        }
    });
}

// Model Analysis Charts
function initModelCharts() {
    // Load metrics from elements dataset attributes
    const metricsContainer = document.getElementById("metrics-data-holder");
    if (!metricsContainer) return;
    
    const metrics = JSON.parse(metricsContainer.textContent);
    const models = ["Decision Tree", "K-Nearest Neighbors", "Random Forest", "XGBoost"];
    
    // Colors
    const colors = [
        "rgba(245, 158, 11, 0.7)",  // Orange for DT
        "rgba(14, 165, 233, 0.7)",  // Blue for KNN
        "rgba(139, 92, 246, 0.7)",  // Purple for RF
        "rgba(13, 148, 136, 0.7)"   // Teal for XGBoost
    ];
    
    const borderColors = [
        "rgb(245, 158, 11)",
        "rgb(14, 165, 233)",
        "rgb(139, 92, 246)",
        "rgb(13, 148, 136)"
    ];

    // Chart 1: Accuracy Comparison Chart
    const ctxAcc = document.getElementById("accuracyChart").getContext("2d");
    const accData = models.map(m => (metrics[m].accuracy * 100).toFixed(2));
    
    new Chart(ctxAcc, {
        type: "bar",
        data: {
            labels: models,
            datasets: [{
                label: "Accuracy (%)",
                data: accData,
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#94a3b8" }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: "#94a3b8" }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) { return ` ${context.parsed.y}%`; }
                    }
                }
            }
        }
    });

    // Chart 2: Metric Suite Comparison Chart (Precision, Recall, F1)
    const ctxMetrics = document.getElementById("metricsComparisonChart").getContext("2d");
    const precisions = models.map(m => (metrics[m].precision * 100).toFixed(1));
    const recalls = models.map(m => (metrics[m].recall * 100).toFixed(1));
    const f1Scores = models.map(m => (metrics[m].f1_score * 100).toFixed(1));
    
    new Chart(ctxMetrics, {
        type: "bar",
        data: {
            labels: models,
            datasets: [
                {
                    label: "Precision (%)",
                    data: precisions,
                    backgroundColor: "rgba(14, 165, 233, 0.6)",
                    borderColor: "rgb(14, 165, 233)",
                    borderWidth: 1.5,
                    borderRadius: 4
                },
                {
                    label: "Recall (%)",
                    data: recalls,
                    backgroundColor: "rgba(16, 185, 129, 0.6)",
                    borderColor: "rgb(16, 185, 129)",
                    borderWidth: 1.5,
                    borderRadius: 4
                },
                {
                    label: "F1-Score (%)",
                    data: f1Scores,
                    backgroundColor: "rgba(245, 158, 11, 0.6)",
                    borderColor: "rgb(245, 158, 11)",
                    borderWidth: 1.5,
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#94a3b8" }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: "#94a3b8" }
                }
            },
            plugins: {
                legend: {
                    labels: { color: "#f8fafc" }
                }
            }
        }
    });
}

// Scenario 2: Command Center Interactive District Simulator
function initControlRoom() {
    const districts = {
        "riverside": {
            name: "Riverside District",
            annual_rainfall: 3100,
            seasonal_rainfall: 1950,
            monthly_rainfall: 580,
            cloud_visibility: 2.2,
            temperature: 22.4,
            humidity: 94,
            atmospheric_pressure: 994,
            wind_speed: 42,
            river_level: 11.2,
            cloud_cover: 92,
            risk_class: "flood-high",
            status: "Normal Operations",
            warning_issued: false,
            resources_allocated: false
        },
        "highland": {
            name: "Highland Valley",
            annual_rainfall: 1100,
            seasonal_rainfall: 520,
            monthly_rainfall: 110,
            cloud_visibility: 8.5,
            temperature: 28.0,
            humidity: 52,
            atmospheric_pressure: 1012,
            wind_speed: 12,
            river_level: 2.4,
            cloud_cover: 25,
            risk_class: "flood-low",
            status: "Normal Operations",
            warning_issued: false,
            resources_allocated: false
        },
        "coastal": {
            name: "Coastal Plains",
            annual_rainfall: 2600,
            seasonal_rainfall: 1680,
            monthly_rainfall: 420,
            cloud_visibility: 3.5,
            temperature: 24.5,
            humidity: 88,
            atmospheric_pressure: 1001,
            wind_speed: 52,
            river_level: 8.9,
            cloud_cover: 82,
            risk_class: "flood-high",
            status: "Normal Operations",
            warning_issued: false,
            resources_allocated: false
        },
        "lakefront": {
            name: "Lakefront Basin",
            annual_rainfall: 2900,
            seasonal_rainfall: 1850,
            monthly_rainfall: 520,
            cloud_visibility: 1.8,
            temperature: 21.0,
            humidity: 96,
            atmospheric_pressure: 998,
            wind_speed: 38,
            river_level: 10.5,
            cloud_cover: 95,
            risk_class: "flood-high",
            status: "Normal Operations",
            warning_issued: false,
            resources_allocated: false
        },
        "dryplains": {
            name: "Dry Plains",
            annual_rainfall: 950,
            seasonal_rainfall: 480,
            monthly_rainfall: 90,
            cloud_visibility: 9.5,
            temperature: 32.5,
            humidity: 42,
            atmospheric_pressure: 1018,
            wind_speed: 8,
            river_level: 1.2,
            cloud_cover: 15,
            risk_class: "flood-low",
            status: "Normal Operations",
            warning_issued: false,
            resources_allocated: false
        }
    };

    let selectedDistrictId = "riverside";
    
    // Select District Elements
    const cards = document.querySelectorAll(".district-card");
    const dNameEl = document.getElementById("detailsName");
    const dRiskEl = document.getElementById("detailsRisk");
    const dStatusEl = document.getElementById("detailsStatus");
    
    // Parameters Elements
    const pAnnual = document.getElementById("pAnnual");
    const pSeasonal = document.getElementById("pSeasonal");
    const pMonthly = document.getElementById("pMonthly");
    const pRiver = document.getElementById("pRiver");
    const pHumidity = document.getElementById("pHumidity");
    const pVisibility = document.getElementById("pVisibility");
    const pCloud = document.getElementById("pCloud");
    
    // Controls Elements
    const btnPredict = document.getElementById("btnRoomPredict");
    const btnEvac = document.getElementById("btnRoomEvac");
    const btnResource = document.getElementById("btnRoomResource");
    const predictionResultEl = document.getElementById("roomPredictionResult");
    
    // Load District Details
    function loadDistrict(id) {
        selectedDistrictId = id;
        const d = districts[id];
        
        // Update active class
        if (cards) {
            cards.forEach(card => card.classList.remove("active"));
            const activeCard = document.querySelector(`[data-district="${id}"]`);
            if (activeCard) activeCard.classList.add("active");
        }
        
        // Update labels
        if (dNameEl) dNameEl.textContent = d.name;
        if (dRiskEl) {
            dRiskEl.textContent = d.risk_class === "flood-high" ? "High Risk (Monsoon Baseline)" : "Low Risk (Stable Baseline)";
            dRiskEl.className = d.risk_class === "flood-high" ? "text-danger fw-bold" : "text-success fw-bold";
        }
        if (dStatusEl) dStatusEl.textContent = d.status;
        
        // Update progress/labels values
        if (pAnnual) pAnnual.textContent = `${d.annual_rainfall} mm`;
        if (pSeasonal) pSeasonal.textContent = `${d.seasonal_rainfall} mm`;
        if (pMonthly) pMonthly.textContent = `${d.monthly_rainfall} mm`;
        if (pRiver) pRiver.textContent = `${d.river_level} m`;
        if (pHumidity) pHumidity.textContent = `${d.humidity}%`;
        if (pVisibility) pVisibility.textContent = `${d.cloud_visibility} km`;
        if (pCloud) pCloud.textContent = `${d.cloud_cover}%`;
        
        // Reset/Update action buttons states
        if (btnEvac) btnEvac.disabled = d.risk_class !== "flood-high";
        if (btnResource) btnResource.disabled = d.risk_class !== "flood-high";
        
        if (btnEvac) {
            if (d.warning_issued) {
                btnEvac.textContent = "Evacuation Alert ACTIVE";
                btnEvac.className = "btn btn-danger w-100 btn-sm";
            } else {
                btnEvac.textContent = "Issue Evacuation Alert";
                btnEvac.className = "btn btn-outline-danger w-100 btn-sm";
            }
        }
        
        if (btnResource) {
            if (d.resources_allocated) {
                btnResource.textContent = "Resources Dispatched";
                btnResource.className = "btn btn-success w-100 btn-sm";
            } else {
                btnResource.textContent = "Allocate Relief Resources";
                btnResource.className = "btn btn-outline-success w-100 btn-sm";
            }
        }
        
        if (predictionResultEl) {
            predictionResultEl.innerHTML = `
                <div class="alert alert-secondary border-0 bg-opacity-10 py-2">
                    <i class="fas fa-microchip me-2"></i> Click "Run Prediction" to check status with XGBoost.
                </div>
            `;
        }
    }
    
    // Add Click listener to cards
    if (cards) {
        cards.forEach(card => {
            card.addEventListener("click", () => {
                const id = card.dataset.district;
                loadDistrict(id);
            });
        });
    }
    
    // Run Prediction
    if (btnPredict) {
        btnPredict.addEventListener("click", async () => {
            const d = districts[selectedDistrictId];
            if (predictionResultEl) {
                predictionResultEl.innerHTML = `
                    <div class="d-flex align-items-center justify-content-center py-2">
                        <div class="spinner-border spinner-border-sm text-info me-2" role="status"></div>
                        <span>Invoking XGBoost Model...</span>
                    </div>
                `;
            }
            
            try {
                const response = await fetch("/api/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        annual_rainfall: d.annual_rainfall,
                        seasonal_rainfall: d.seasonal_rainfall,
                        monthly_rainfall: d.monthly_rainfall,
                        cloud_visibility: d.cloud_visibility,
                        temperature: d.temperature,
                        humidity: d.humidity,
                        atmospheric_pressure: d.atmospheric_pressure,
                        wind_speed: d.wind_speed,
                        river_level: d.river_level,
                        cloud_cover: d.cloud_cover
                    })
                });
                
                const result = await response.json();
                
                if (result.error) {
                    if (predictionResultEl) predictionResultEl.innerHTML = `<div class="text-danger small">Error: ${result.error}</div>`;
                    return;
                }
                
                const isFlood = result.prediction === 1;
                const badgeClass = isFlood ? "bg-danger" : "bg-success";
                const iconClass = isFlood ? "fa-exclamation-triangle" : "fa-check-circle";
                
                if (predictionResultEl) {
                    predictionResultEl.innerHTML = `
                        <div class="glass-card mt-2 border-0 bg-opacity-20 p-3">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <span class="badge ${badgeClass} p-2"><i class="fas ${iconClass} me-1"></i> ${result.status}</span>
                                <strong class="text-white">${result.confidence}% Confidence</strong>
                            </div>
                            <p class="small text-secondary mb-0">Model Score: ${result.probability} (Threshold: ${result.threshold})</p>
                        </div>
                    `;
                }
                
                // Sync status if prediction changes
                if (isFlood && !d.warning_issued && !d.resources_allocated) {
                    d.status = "Warning Pending";
                    if (dStatusEl) dStatusEl.textContent = d.status;
                }
                
            } catch (err) {
                if (predictionResultEl) predictionResultEl.innerHTML = `<div class="text-danger small">Prediction API unavailable.</div>`;
            }
        });
    }
    
    // Issue Evacuation Alert
    if (btnEvac) {
        btnEvac.addEventListener("click", () => {
            const d = districts[selectedDistrictId];
            d.warning_issued = !d.warning_issued;
            
            if (d.warning_issued) {
                d.status = d.resources_allocated ? "Active Evacuation & Relief" : "Evacuation Protocol Active";
                btnEvac.textContent = "Evacuation Alert ACTIVE";
                btnEvac.className = "btn btn-danger w-100 btn-sm";
                alert(`🚨 EMERGENCY NOTICE: Evacuation warnings successfully transmitted to all emergency frequencies and residents in ${d.name}!`);
            } else {
                d.status = d.resources_allocated ? "Relief Operations Active" : "Normal Operations";
                btnEvac.textContent = "Issue Evacuation Alert";
                btnEvac.className = "btn btn-outline-danger w-100 btn-sm";
            }
            
            if (dStatusEl) dStatusEl.textContent = d.status;
            
            // Update badge status on the left card
            const cardBadge = document.querySelector(`[data-district="${selectedDistrictId}"] .badge`);
            if (cardBadge) {
                if (d.warning_issued || d.resources_allocated) {
                    cardBadge.textContent = "ACTIVE DISASTER";
                    cardBadge.className = "badge bg-danger float-end";
                } else {
                    cardBadge.textContent = "HIGH RISK";
                    cardBadge.className = "badge bg-warning float-end text-dark";
                }
            }
        });
    }
    
    // Allocate Resources
    if (btnResource) {
        btnResource.addEventListener("click", () => {
            const d = districts[selectedDistrictId];
            d.resources_allocated = !d.resources_allocated;
            
            if (d.resources_allocated) {
                d.status = d.warning_issued ? "Active Evacuation & Relief" : "Relief Operations Active";
                btnResource.textContent = "Resources Dispatched";
                btnResource.className = "btn btn-success w-100 btn-sm";
                alert(`🚚 RESOURCE DEPLOYMENT: Emergency personnel, 5 rescue boats, 200 food packets, and medical support packages dispatched to ${d.name}.`);
            } else {
                d.status = d.warning_issued ? "Evacuation Protocol Active" : "Normal Operations";
                btnResource.textContent = "Allocate Relief Resources";
                btnResource.className = "btn btn-outline-success w-100 btn-sm";
            }
            
            if (dStatusEl) dStatusEl.textContent = d.status;
        });
    }
    
    // Load Riverside by default
    loadDistrict("riverside");
}
