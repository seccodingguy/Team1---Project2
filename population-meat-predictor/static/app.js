// app.js

// Global variables
let populationChart = null;
let meatChart = null;
let currentCharts = {};



// DOM Elements
const form = document.getElementById('prediction-form');
const resultsSection = document.getElementById('results-section');
const loadingIndicator = document.getElementById('loading-indicator');
const errorMessage = document.getElementById('error-message');
const resultsContent = document.getElementById('results-content');

// Utility Functions
const formatNumber = (value) => {
    if (Math.abs(value) >= 1000000) {
        return (value / 1000000).toFixed(2) + 'M';
    } else if (Math.abs(value) >= 1000) {
        return (value / 1000).toFixed(2) + 'K';
    }
    return value.toFixed(2);
};

const formatMetricName = (key) => {
    return key.split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
};

// Function to destroy all existing charts
const destroyAllCharts = () => {
    const canvas_elements = document.getElementsByTagName("canvas");

    for (let i = 0; i < canvas_elements.length; i++) {
        if(document.getElementById(canvas_elements[i].id).getContext('2d') !== undefined) {
            const ctx = document.getElementById(canvas_elements[i].id).getContext('2d');
            if (ctx instanceof Chart) {
                alert(ctx.id);
                ctx.destroy();
            }
            
        }
      }
      
};




// UI State Management
const showLoading = () => {
    resultsSection.classList.remove('hidden');
    loadingIndicator.classList.remove('hidden');
    errorMessage.classList.add('hidden');
    resultsContent.classList.add('hidden');
};

const showError = (message) => {
    loadingIndicator.classList.add('hidden');
    errorMessage.classList.remove('hidden');
    resultsContent.classList.add('hidden');
    document.getElementById('error-text').textContent = message;
};


// Chart Creation Functions
const createCharts = (data) => {
    const years = data.linear_regression.years;
    const models = ['linear_regression', 'lasso_regression', 'ridge_regression'];
    const colors = {
        linear_regression: 'rgb(59, 130, 246)', // blue
        lasso_regression: 'rgb(16, 185, 129)', // green
        ridge_regression: 'rgb(249, 115, 22)' // orange
    };

    // Destroy existing charts
    if (populationChart) populationChart.destroy();
    if (meatChart) meatChart.destroy();
    
    // Create population chart
    const popCtx = document.getElementById('population-chart').getContext('2d');
    
    new Chart(popCtx, {
        type: 'line',
        data: {
            labels: years,
            datasets: models.map(model => ({
                label: model.split('_')[0].charAt(0).toUpperCase() + model.split('_')[0].slice(1),
                data: data[model].population_predictions,
                borderColor: colors[model],
                tension: 0.1,
                fill: false
            }))
        },
        options: getChartOptions('Population Predictions')
    });

    // Create meat consumption chart
    const meatCtx = document.getElementById('meat-chart').getContext('2d');
   
    new Chart(meatCtx, {
        type: 'line',
        data: {
            labels: years,
            datasets: models.map(model => ({
                label: model.split('_')[0].charAt(0).toUpperCase() + model.split('_')[0].slice(1),
                data: data[model].meat_predictions,
                borderColor: colors[model],
                tension: 0.1,
                fill: false
            }))
        },
        options: getChartOptions('Meat Consumption Predictions')
    });
};

const getChartOptions = (title) => {
    return {
        responsive: true,
        interaction: {
            intersect: false,
            mode: 'index'
        },
        plugins: {
            title: {
                display: true,
                text: title
            },
            legend: {
                position: 'bottom'
            }
        },
        scales: {
            y: {
                beginAtZero: false
            }
        }
    };
};

const createGridSearchPlots = (data) => {
    createParameterPlot('ridge', data.ridge_grid_search);
    createParameterPlot('lasso', data.lasso_grid_search);
    updateBestParameters(data);
};

const createParameterPlot = (modelType, gridData) => {
    const ctx = document.getElementById(`${modelType}-params-chart`).getContext('2d');
   
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: gridData.alphas.map(a => `α=${a.toExponential(2)}`),
            datasets: [
                {
                    label: 'Population Score',
                    data: gridData.population_scores,
                    borderColor: 'rgb(59, 130, 246)',
                    tension: 0.1,
                    yAxisID: 'y'
                },
                {
                    label: 'Meat Consumption Score',
                    data: gridData.meat_scores,
                    borderColor: 'rgb(220, 38, 38)',
                    tension: 0.1,
                    yAxisID: 'y'
                }
            ]
        },
        options: getGridSearchOptions(`${modelType.charAt(0).toUpperCase() + modelType.slice(1)} Regression Cross-Validation Scores`)
    });
    

};

const getGridSearchOptions = (title) => {
    return {
        responsive: true,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            title: {
                display: true,
                text: title
            }
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Alpha Values'
                }
            },
            y: {
                type: 'linear',
                display: true,
                position: 'left',
                title: {
                    display: true,
                    text: 'Cross-Validation Score'
                }
            }
        }
    };
};

const updateBestParameters = (data) => {
    ['ridge', 'lasso'].forEach(model => {
        document.getElementById(`${model}-best-params`).innerHTML = `
            <div class="bg-white p-3 rounded-lg">
                <h4 class="font-medium text-gray-700 mb-2">Best Parameters:</h4>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <span class="font-medium">Population:</span>
                        <span class="ml-2">α=${data[`${model}_grid_search`].best_params.population.toExponential(2)}</span>
                    </div>
                    <div>
                        <span class="font-medium">Meat Consumption:</span>
                        <span class="ml-2">α=${data[`${model}_grid_search`].best_params.meat.toExponential(2)}</span>
                    </div>
                </div>
            </div>
        `;
    });
};

const createScatterPlots = (data) => {
    const models = ['linear', 'lasso', 'ridge'];
    const metrics = ['population', 'meat'];
    
    models.forEach(model => {
        metrics.forEach(metric => {
            createScatterPlot(model, metric, data);
        });
    });
};

const createScatterPlot = (model, metric, data) => {
    const modelKey = `${model}_regression`;
    const ctx = document.getElementById(`${model}-${metric}-scatter`).getContext('2d');
   
    if (!data[modelKey] || 
        !data[modelKey][`${metric}_actual`] || 
        !data[modelKey][`${metric}_predicted`] ||
        !data[modelKey][`${metric}_bagged`] ||
        !data[modelKey][`${metric}_noise`] ) {
        console.error(`Missing data for ${model} ${metric}`);
        return;
    }
    
    const actualValues = data[modelKey][`${metric}_actual`];
    const predictedValues = data[modelKey][`${metric}_predicted`];
    const baggedValues = data[modelKey][`${metric}_bagged`];
    const noiseValues = data[modelKey][`${metric}_noise`];
    const minVal = Math.min(...actualValues);
    const maxVal = Math.max(...actualValues);
    const predMinVal = Math.min(...predictedValues);
    const predMaxVal = Math.max(...predictedValues);

    new Chart(ctx, {
        type: 'scatter',
        data: getScatterData(actualValues, predictedValues, minVal, maxVal),
        options: getScatterOptions()
    });

    const ctxBagged = document.getElementById(`${model}-${metric}-bagged-scatter`).getContext('2d');

    new Chart(ctxBagged,{
        type: 'scatter',
        data: getBaggedScatterData(predictedValues, baggedValues, predMinVal, predMaxVal),
        options: getScatterOptions()
    });

    const ctxNoise = document.getElementById(`${model}-${metric}-noise-scatter`).getContext('2d');

    new Chart(ctxNoise,{
        type: 'scatter',
        data: getNoiseScatterData(predictedValues, noiseValues, predMinVal, predMaxVal),
        options: getScatterOptions()
    });
};

const getNoiseScatterData = (actualValues, noiseValues, minVal, maxVal) => {
    return {
        datasets: [
            {
                label: 'With Gaussian Noise Predictions',
                data: actualValues.map((actual, i) => ({
                    x: actual,
                    y: noiseValues[i]
                })),
                backgroundColor: 'rgba(246, 165, 59, 0.79)',
                borderColor: 'rgba(59, 153, 246, 0.8)',
                pointRadius: 4
            },
            {
                label: 'Original Prediction',
                data: [
                    { x: minVal, y: minVal },
                    { x: maxVal, y: maxVal }
                ],
                type: 'line',
                borderColor: 'rgba(220, 38, 38, 0.5)',
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false
            }
        ]
    };
}

const getBaggedScatterData = (actualValues, baggedValues, minVal, maxVal) => {
    return {
        datasets: [
            {
                label: 'Bagging Predictions',
                data: actualValues.map((actual, i) => ({
                    x: actual,
                    y: baggedValues[i]
                })),
                backgroundColor: 'rgba(168, 246, 59, 0.79)',
                borderColor: 'rgba(187, 59, 246, 0.8)',
                pointRadius: 4
            },
            {
                label: 'Original Predictions',
                data: [
                    { x: minVal, y: minVal },
                    { x: maxVal, y: maxVal }
                ],
                type: 'line',
                borderColor: 'rgba(220, 38, 38, 0.5)',
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false
            }
        ]
    };
}

const getScatterData = (actualValues, predictedValues, minVal, maxVal) => {
    return {
        datasets: [
            {
                label: 'Test Set Predictions',
                data: actualValues.map((actual, i) => ({
                    x: actual,
                    y: predictedValues[i]
                })),
                backgroundColor: 'rgba(59, 130, 246, 0.5)',
                borderColor: 'rgba(59, 130, 246, 0.8)',
                pointRadius: 4
            },
            {
                label: 'Perfect Prediction',
                data: [
                    { x: minVal, y: minVal },
                    { x: maxVal, y: maxVal }
                ],
                type: 'line',
                borderColor: 'rgba(220, 38, 38, 0.5)',
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false
            }
        ]
    };
};

const getScatterOptions = () => {
    return {
        responsive: true,
        aspectRatio: 1,
        plugins: {
            title: { display: false },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        if (context.datasetIndex === 0) {
                            return [
                                `Actual: ${formatNumber(context.parsed.x)}`,
                                `Predicted: ${formatNumber(context.parsed.y)}`
                            ];
                        }
                        return '';
                    }
                }
            }
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Actual Values'
                },
                ticks: {
                    callback: value => formatNumber(value)
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'Predicted Values'
                },
                ticks: {
                    callback: value => formatNumber(value)
                }
            }
        }
    };
};

const updateMetrics = (data) => {
    const models = ['linear', 'lasso', 'ridge'];
    models.forEach(model => {
        const modelData = data[`${model}_regression`];
        const metricsDiv = document.getElementById(`${model}-metrics`);
        
        updateMetricSection(metricsDiv, '.metrics-population', 'Population Metrics', modelData.metrics_population);
        updateMetricSection(metricsDiv, '.metrics-meat', 'Meat Consumption Metrics', modelData.metrics_meat);
    });
};

const updateMetricSection = (container, selector, title, metrics) => {
    const section = container.querySelector(selector);
    section.innerHTML = `
        <h4 class="font-medium text-gray-600">${title}</h4>
        ${Object.entries(metrics)
            .map(([key, value]) => `
                <div class="text-sm">
                    <span class="font-medium">${formatMetricName(key)}:</span>
                    <span class="ml-2">${formatNumber(value)}</span>
                </div>
            `).join('')}
    `;
};

const updateTable = (data) => {
    const tableBody = document.getElementById('predictions-table-body');
    const models = ['linear', 'lasso', 'ridge'];
    
    const tableContent = data.linear_regression.years.map((year, yearIndex) => {
        return models.map((model, modelIndex) => {
            const modelData = data[`${model}_regression`];
            return `
                <tr class="${yearIndex > 0 && modelIndex === 0 ? 'border-t border-gray-200' : ''}">
                    ${modelIndex === 0 ? `
                        <td rowspan="3" class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 font-medium">
                            ${year}
                        </td>
                    ` : ''}
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        ${model.charAt(0).toUpperCase() + model.slice(1)}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        ${formatNumber(modelData.population_predictions[yearIndex])}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        ${formatNumber(modelData.meat_predictions[yearIndex])}
                    </td>
                </tr>
            `;
        }).join('');
    }).join('');

    tableBody.innerHTML = tableContent;
};

const loadTemplates = async () => {
    try {
        const [plotsResponse, resultsResponse] = await Promise.all([
            fetch('/plots'),
            fetch('/results')
        ]);

        if (!plotsResponse.ok || !resultsResponse.ok) {
            throw new Error('Failed to load templates');
        }

        const plotsHtml = await plotsResponse.text();
        const resultsHtml = await resultsResponse.text();

        document.getElementById('plots-container').innerHTML = plotsHtml;
        document.getElementById('results-container').innerHTML = resultsHtml;

        return true;
    } catch (error) {
        console.error('Error loading templates:', error);
        return false;
    }
};

// Modify showResults function
const showResults = async (data) => {
    if (!data) {
        showError('No data received from the server');
        return;
    }

    try {
        // Show the results container
        loadingIndicator.classList.add('hidden');
        errorMessage.classList.add('hidden');
        resultsContent.classList.remove('hidden');

        // Load templates
        const templatesLoaded = await loadTemplates();
        if (!templatesLoaded) {
            throw new Error('Failed to load result templates');
        }

        // Small delay to ensure DOM is updated
        setTimeout(() => {
            try {
                //destroyAllCharts();
                createCharts(data);
                updateMetrics(data);
                updateTable(data);
                if (data.ridge_grid_search && data.lasso_grid_search) {
                    createGridSearchPlots(data);
                }
                createScatterPlots(data);
            } catch (error) {
                console.error('Error creating charts:', error);
                showError('Error creating charts: ' + error.message);
            }
        }, 100);
    } catch (error) {
        console.error('Error in showResults:', error);
        showError('Error displaying results: ' + error.message);
    }
};

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        showLoading();
        

        const formData = {
            country: document.getElementById('country').value,
            meat_category: document.getElementById('meat_category').value,
            start_year: parseInt(document.getElementById('start_year').value),
            future_years: parseInt(document.getElementById('future_years').value)
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'An error occurred while generating predictions');
            }

            showResults(data);
        } catch (error) {
            showError(error.message);
        }
    });
});