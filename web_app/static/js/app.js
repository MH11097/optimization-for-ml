class VisualizationApp {
  constructor() {
    this.algorithms = [];
    this.currentAlgorithm = null;
    this.currentParameterRanges = {};
    this.currentGroupedSetups = {};
    this.currentSetup = null;
    this.comparisonMode = false;
    this.selectedSetups = [];
    this.charts = {};

    // DOM elements
    this.elements = {};

    // Bind methods
    this.handleAlgorithmChange = this.handleAlgorithmChange.bind(this);
    this.handleParameterGroupChange =
      this.handleParameterGroupChange.bind(this);
    this.handleParameterChange = this.handleParameterChange.bind(this);
    this.handleChartTypeChange = this.handleChartTypeChange.bind(this);
    this.handleComparisonModeChange =
      this.handleComparisonModeChange.bind(this);
  }

  init() {
    this.initElements();
    this.bindEvents();
    this.loadAlgorithms();
    this.showNoDataMessage();
  }

  initElements() {
    this.elements = {
      algorithmSelect: document.getElementById("algorithmSelect"),
      parameterGroupSelect: document.getElementById("parameterGroupSelect"),
      parameterGroupContainer: document.getElementById(
        "parameterGroupContainer"
      ),
      parametersContainer: document.getElementById("parametersContainer"),
      slidersContainer: document.getElementById("slidersContainer"),
      setupInfo: document.getElementById("setupInfo"),
      setupDetails: document.getElementById("setupDetails"),
      statsCard: document.getElementById("statsCard"),
      statsContent: document.getElementById("statsContent"),
      comparisonMode: document.getElementById("comparisonMode"),
      comparisonSetups: document.getElementById("comparisonSetups"),
      selectedSetupsList: document.getElementById("selectedSetupsList"),
      addCurrentSetup: document.getElementById("addCurrentSetup"),
      clearComparison: document.getElementById("clearComparison"),
      chartTypeButtons: document.querySelectorAll('input[name="chartType"]'),
      charts2D: document.getElementById("charts2D"),
      charts3D: document.getElementById("charts3D"),
      loadingIndicator: document.getElementById("loadingIndicator"),
      errorMessage: document.getElementById("errorMessage"),
      errorText: document.getElementById("errorText"),
      noDataMessage: document.getElementById("noDataMessage"),
    };
  }

  bindEvents() {
    this.elements.algorithmSelect.addEventListener(
      "change",
      this.handleAlgorithmChange
    );
    this.elements.parameterGroupSelect.addEventListener(
      "change",
      this.handleParameterGroupChange
    );
    this.elements.comparisonMode.addEventListener(
      "change",
      this.handleComparisonModeChange
    );
    this.elements.addCurrentSetup.addEventListener("click", () =>
      this.addCurrentSetupToComparison()
    );
    this.elements.clearComparison.addEventListener("click", () =>
      this.clearComparison()
    );

    this.elements.chartTypeButtons.forEach((button) => {
      button.addEventListener("change", this.handleChartTypeChange);
    });
  }

  async loadAlgorithms() {
    try {
      this.showLoading();
      const response = await fetch("/api/algorithms");
      const data = await response.json();

      if (data.success) {
        this.algorithms = data.algorithms;
        this.populateAlgorithmSelect();
        this.hideLoading();
      } else {
        this.showError("Failed to load algorithms: " + data.error);
      }
    } catch (error) {
      this.showError("Network error loading algorithms: " + error.message);
    }
  }

  populateAlgorithmSelect() {
    this.elements.algorithmSelect.innerHTML =
      '<option value="">Select Algorithm...</option>';

    this.algorithms.forEach((algorithm) => {
      const option = document.createElement("option");
      option.value = algorithm;
      option.textContent = algorithm
        .replace(/_/g, " ")
        .replace(/\b\w/g, (l) => l.toUpperCase());
      this.elements.algorithmSelect.appendChild(option);
    });
  }

  async handleAlgorithmChange() {
    const selectedAlgorithm = this.elements.algorithmSelect.value;

    if (!selectedAlgorithm) {
      this.resetUI();
      return;
    }

    try {
      this.showLoading();
      this.currentAlgorithm = selectedAlgorithm;

      // Load parameter ranges and grouped setups
      await Promise.all([
        this.loadParameterRanges(selectedAlgorithm),
        this.loadGroupedSetups(selectedAlgorithm),
      ]);

      this.populateParameterGroupSelect();
      this.hideLoading();
    } catch (error) {
      this.showError("Error loading algorithm data: " + error.message);
    }
  }

  async loadParameterRanges(algorithm) {
    const response = await fetch(
      `/api/algorithms/${algorithm}/parameter-ranges`
    );
    const data = await response.json();

    if (data.success) {
      this.currentParameterRanges = data.parameter_ranges;
    } else {
      throw new Error(data.error);
    }
  }

  async loadGroupedSetups(algorithm) {
    const response = await fetch(`/api/algorithms/${algorithm}/grouped-setups`);
    const data = await response.json();

    if (data.success) {
      this.currentGroupedSetups = data.grouped_setups;
    } else {
      throw new Error(data.error);
    }
  }

  populateParameterGroupSelect() {
    this.elements.parameterGroupSelect.innerHTML =
      '<option value="">Select Group...</option>';

    Object.keys(this.currentGroupedSetups).forEach((group) => {
      const option = document.createElement("option");
      option.value = group;
      option.textContent = group
        .replace(/_/g, " ")
        .replace(/\b\w/g, (l) => l.toUpperCase());
      this.elements.parameterGroupSelect.appendChild(option);
    });

    this.elements.parameterGroupContainer.style.display = "block";
  }

  handleParameterGroupChange() {
    const selectedGroup = this.elements.parameterGroupSelect.value;

    if (!selectedGroup) {
      this.elements.parametersContainer.style.display = "none";
      return;
    }

    this.createParameterSliders(selectedGroup);
    this.elements.parametersContainer.style.display = "block";
  }

  createParameterSliders(group) {
    const setups = this.currentGroupedSetups[group];
    if (!setups || setups.length === 0) return;

    // Get common parameters across all setups in this group
    const commonParams = this.getCommonParameters(setups);

    this.elements.slidersContainer.innerHTML = "";

    Object.keys(commonParams).forEach((param) => {
      if (this.currentParameterRanges[param]) {
        this.createSlider(param, this.currentParameterRanges[param]);
      }
    });

    // Initialize with first setup
    if (setups.length > 0) {
      this.loadSetupData(setups[0]);
    }
  }

  getCommonParameters(setups) {
    if (setups.length === 0) return {};

    const commonParams = {};
    const firstSetup = setups[0].parsed_parameters;

    Object.keys(firstSetup).forEach((param) => {
      if (typeof firstSetup[param] === "number") {
        commonParams[param] = new Set();
        setups.forEach((setup) => {
          if (setup.parsed_parameters[param] !== undefined) {
            commonParams[param].add(setup.parsed_parameters[param]);
          }
        });
      }
    });

    return commonParams;
  }

  createSlider(paramName, paramRange) {
    const sliderDiv = document.createElement("div");
    sliderDiv.className = "parameter-slider";

    const currentValue = paramRange.values[0]; // Default to first value

    sliderDiv.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <label class="parameter-label">${paramName.replace(
                  /_/g,
                  " "
                )}</label>
                <span class="slider-value" id="${paramName}_value">${currentValue}</span>
            </div>
            <div class="slider-container">
                <input type="range" 
                       class="form-range" 
                       id="${paramName}_slider"
                       min="0" 
                       max="${paramRange.values.length - 1}" 
                       step="1" 
                       value="0">
            </div>
        `;

    this.elements.slidersContainer.appendChild(sliderDiv);

    // Bind slider event
    const slider = document.getElementById(`${paramName}_slider`);
    const valueDisplay = document.getElementById(`${paramName}_value`);

    slider.addEventListener("input", (e) => {
      const index = parseInt(e.target.value);
      const value = paramRange.values[index];
      valueDisplay.textContent = value;
      this.handleParameterChange();
    });
  }

  async handleParameterChange() {
    if (!this.currentAlgorithm) return;

    // Get current parameter values from sliders
    const currentParams = this.getCurrentParameterValues();

    // Find matching setup
    try {
      const setup = await this.findMatchingSetup(currentParams);
      if (setup) {
        this.loadSetupData(setup);
      }
    } catch (error) {
      console.error("Error finding matching setup:", error);
    }
  }

  getCurrentParameterValues() {
    const params = {};
    const sliders =
      this.elements.slidersContainer.querySelectorAll(".form-range");

    sliders.forEach((slider) => {
      const paramName = slider.id.replace("_slider", "");
      const index = parseInt(slider.value);
      const paramRange = this.currentParameterRanges[paramName];
      if (paramRange) {
        params[paramName] = paramRange.values[index];
      }
    });

    return params;
  }

  async findMatchingSetup(targetParams) {
    const queryString = Object.keys(targetParams)
      .map((key) => `${key}=${targetParams[key]}`)
      .join("&");

    const response = await fetch(
      `/api/algorithms/${this.currentAlgorithm}/setup-by-params?${queryString}`
    );
    const data = await response.json();

    if (data.success) {
      return data.setup;
    }

    return null;
  }

  async loadSetupData(setup) {
    this.currentSetup = setup;
    this.updateSetupInfo(setup);
    this.updateStatistics(setup);

    if (this.comparisonMode) {
      this.updateComparisonCharts();
    } else {
      this.updateSingleSetupCharts(setup);
    }
  }

  updateSetupInfo(setup) {
    const details = Object.keys(setup.parsed_parameters)
      .filter((key) => typeof setup.parsed_parameters[key] === "number")
      .map(
        (key) => `
                <div class="setup-parameter">
                    <span>${key.replace(/_/g, " ")}:</span>
                    <span>${setup.parsed_parameters[key]}</span>
                </div>
            `
      )
      .join("");

    this.elements.setupDetails.innerHTML = `
            <div class="mb-2"><strong>${setup.setup_name}</strong></div>
            ${details}
        `;

    this.elements.setupInfo.style.display = "block";
  }

  updateStatistics(setup) {
    const results = setup.results;
    const training = results.training_results;

    const stats = [
      { label: "Final Loss", value: training.final_loss?.toFixed(6) || "N/A" },
      { label: "Iterations", value: training.total_iterations || "N/A" },
      { label: "Converged", value: training.converged ? "Yes" : "No" },
      {
        label: "Training Time",
        value: training.training_time
          ? `${training.training_time.toFixed(2)}s`
          : "N/A",
      },
      {
        label: "Gradient Norm",
        value: training.final_gradient_norm?.toFixed(6) || "N/A",
      },
    ];

    this.elements.statsContent.innerHTML = stats
      .map(
        (stat) => `
            <div class="stat-item">
                <span class="stat-label">${stat.label}</span>
                <span class="stat-value">${stat.value}</span>
            </div>
        `
      )
      .join("");

    this.elements.statsCard.style.display = "block";
  }

  async updateSingleSetupCharts(setup) {
    this.hideError();
    this.hideNoDataMessage();

    if (!setup.has_history) {
      this.showNoDataMessage();
      return;
    }

    try {
      // Load training history
      const historyPath = setup.setup_path.replace(/.*03_algorithms[\\/]/, "");
      const response = await fetch(`/api/setup/${historyPath}/history`);
      const data = await response.json();

      if (data.success) {
        this.create2DCharts([{ setup, history: data.history }]);
        this.create3DCharts([{ setup, history: data.history }]);
      } else {
        this.showError("Failed to load training history: " + data.error);
      }
    } catch (error) {
      this.showError("Error loading charts: " + error.message);
    }
  }

  create2DCharts(dataList) {
    // Loss convergence chart
    const lossTraces = dataList.map((item, index) => ({
      x: item.history.map((h) => h.iteration),
      y: item.history.map((h) => h.loss),
      type: "scatter",
      mode: "lines",
      name: item.setup.setup_name,
      line: { color: this.getColor(index) },
    }));

    Plotly.newPlot("lossChart", lossTraces, {
      title: "Loss Convergence",
      xaxis: { title: "Iteration" },
      yaxis: { title: "Loss", type: "log" },
      margin: { l: 50, r: 50, t: 50, b: 50 },
    });

    // Gradient norm chart
    const gradientTraces = dataList.map((item, index) => ({
      x: item.history.map((h) => h.iteration),
      y: item.history.map((h) => h.gradient_norm),
      type: "scatter",
      mode: "lines",
      name: item.setup.setup_name,
      line: { color: this.getColor(index) },
    }));

    Plotly.newPlot("gradientChart", gradientTraces, {
      title: "Gradient Norm",
      xaxis: { title: "Iteration" },
      yaxis: { title: "Gradient Norm", type: "log" },
      margin: { l: 50, r: 50, t: 50, b: 50 },
    });

    // Comparison chart (loss vs gradient norm)
    const comparisonTraces = dataList.map((item, index) => ({
      x: item.history.map((h) => h.loss),
      y: item.history.map((h) => h.gradient_norm),
      type: "scatter",
      mode: "markers",
      name: item.setup.setup_name,
      marker: { color: this.getColor(index), size: 4 },
    }));

    Plotly.newPlot("comparisonChart", comparisonTraces, {
      title: "Loss vs Gradient Norm",
      xaxis: { title: "Loss", type: "log" },
      yaxis: { title: "Gradient Norm", type: "log" },
      margin: { l: 50, r: 50, t: 50, b: 50 },
    });
  }

  create3DCharts(dataList) {
    if (dataList.length === 0) return;

    // Parameter space 3D (if we have multiple setups with different parameters)
    if (dataList.length > 1) {
      const paramKeys = Object.keys(dataList[0].setup.parsed_parameters).filter(
        (key) => typeof dataList[0].setup.parsed_parameters[key] === "number"
      );

      if (paramKeys.length >= 2) {
        const trace3D = {
          x: dataList.map((item) => item.setup.parsed_parameters[paramKeys[0]]),
          y: dataList.map((item) => item.setup.parsed_parameters[paramKeys[1]]),
          z: dataList.map(
            (item) => item.setup.results.training_results.final_loss
          ),
          type: "scatter3d",
          mode: "markers",
          marker: {
            size: 8,
            color: dataList.map(
              (item) => item.setup.results.training_results.final_loss
            ),
            colorscale: "Viridis",
            showscale: true,
          },
          text: dataList.map((item) => item.setup.setup_name),
        };

        Plotly.newPlot("parameterSpace3D", [trace3D], {
          title: "Parameter Space",
          scene: {
            xaxis: { title: paramKeys[0] },
            yaxis: { title: paramKeys[1] },
            zaxis: { title: "Final Loss" },
          },
          margin: { l: 0, r: 0, t: 50, b: 0 },
        });
      }
    }

    // Convergence surface (iteration vs loss vs gradient norm)
    const firstItem = dataList[0];
    const surfaceTrace = {
      x: firstItem.history.map((h) => h.iteration),
      y: firstItem.history.map((h) => h.loss),
      z: firstItem.history.map((h) => h.gradient_norm),
      type: "scatter3d",
      mode: "lines+markers",
      marker: {
        size: 3,
        color: firstItem.history.map((h, i) => i),
        colorscale: "Plasma",
        showscale: true,
      },
      line: {
        color: "rgb(100, 100, 100)",
        width: 2,
      },
    };

    Plotly.newPlot("convergenceSurface3D", [surfaceTrace], {
      title: "Convergence Trajectory",
      scene: {
        xaxis: { title: "Iteration" },
        yaxis: { title: "Loss" },
        zaxis: { title: "Gradient Norm" },
      },
      margin: { l: 0, r: 0, t: 50, b: 0 },
    });
  }

  getColor(index) {
    const colors = [
      "#1f77b4",
      "#ff7f0e",
      "#2ca02c",
      "#d62728",
      "#9467bd",
      "#8c564b",
      "#e377c2",
      "#7f7f7f",
      "#bcbd22",
      "#17becf",
    ];
    return colors[index % colors.length];
  }

  handleChartTypeChange(e) {
    if (e.target.value === "2d") {
      this.elements.charts2D.style.display = "block";
      this.elements.charts3D.style.display = "none";
    } else {
      this.elements.charts2D.style.display = "none";
      this.elements.charts3D.style.display = "block";
    }
  }

  handleComparisonModeChange() {
    this.comparisonMode = this.elements.comparisonMode.checked;

    if (this.comparisonMode) {
      this.elements.comparisonSetups.style.display = "block";
      this.updateComparisonCharts();
    } else {
      this.elements.comparisonSetups.style.display = "none";
      if (this.currentSetup) {
        this.updateSingleSetupCharts(this.currentSetup);
      }
    }
  }

  addCurrentSetupToComparison() {
    if (
      this.currentSetup &&
      !this.selectedSetups.find(
        (s) => s.setup_name === this.currentSetup.setup_name
      )
    ) {
      this.selectedSetups.push(this.currentSetup);
      this.updateSelectedSetupsList();
      this.updateComparisonCharts();
    }
  }

  clearComparison() {
    this.selectedSetups = [];
    this.updateSelectedSetupsList();
    this.clearCharts();
  }

  updateSelectedSetupsList() {
    this.elements.selectedSetupsList.innerHTML = this.selectedSetups
      .map(
        (setup, index) => `
            <div class="selected-setup">
                <span>${setup.setup_name}</span>
                <span class="remove-setup" onclick="app.removeSetupFromComparison(${index})">✕</span>
            </div>
        `
      )
      .join("");
  }

  removeSetupFromComparison(index) {
    this.selectedSetups.splice(index, 1);
    this.updateSelectedSetupsList();
    this.updateComparisonCharts();
  }

  async updateComparisonCharts() {
    if (this.selectedSetups.length === 0) {
      this.clearCharts();
      return;
    }

    try {
      const dataList = [];

      for (const setup of this.selectedSetups) {
        if (setup.has_history) {
          const historyPath = setup.setup_path.replace(
            /.*03_algorithms[\\/]/,
            ""
          );
          const response = await fetch(`/api/setup/${historyPath}/history`);
          const data = await response.json();

          if (data.success) {
            dataList.push({ setup, history: data.history });
          }
        }
      }

      if (dataList.length > 0) {
        this.create2DCharts(dataList);
        this.create3DCharts(dataList);
        this.hideNoDataMessage();
      } else {
        this.showNoDataMessage();
      }
    } catch (error) {
      this.showError("Error loading comparison charts: " + error.message);
    }
  }

  clearCharts() {
    [
      "lossChart",
      "gradientChart",
      "comparisonChart",
      "parameterSpace3D",
      "convergenceSurface3D",
    ].forEach((chartId) => {
      Plotly.purge(chartId);
    });
  }

  resetUI() {
    this.currentAlgorithm = null;
    this.currentParameterRanges = {};
    this.currentGroupedSetups = {};
    this.currentSetup = null;

    this.elements.parameterGroupContainer.style.display = "none";
    this.elements.parametersContainer.style.display = "none";
    this.elements.setupInfo.style.display = "none";
    this.elements.statsCard.style.display = "none";

    this.clearCharts();
    this.showNoDataMessage();
  }

  showLoading() {
    this.elements.loadingIndicator.style.display = "block";
    this.hideError();
    this.hideNoDataMessage();
  }

  hideLoading() {
    this.elements.loadingIndicator.style.display = "none";
  }

  showError(message) {
    this.elements.errorText.textContent = message;
    this.elements.errorMessage.style.display = "block";
    this.hideLoading();
    this.hideNoDataMessage();
  }

  hideError() {
    this.elements.errorMessage.style.display = "none";
  }

  showNoDataMessage() {
    this.elements.noDataMessage.style.display = "block";
    this.hideError();
    this.hideLoading();
  }

  hideNoDataMessage() {
    this.elements.noDataMessage.style.display = "none";
  }
}
// Global app instance
let app;
