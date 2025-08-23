/* Master Optimization Framework - JavaScript */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initializeTooltips();
    initializeAnimations();
    initializeFormValidation();
    initializeCharts();
    initializeFileUploads();
    
    console.log('Master Optimization Framework initialized');
});

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize entrance animations
 */
function initializeAnimations() {
    // Animate cards on load
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
    
    // Animate navigation items
    const navItems = document.querySelectorAll('.nav-link');
    navItems.forEach((item, index) => {
        item.style.opacity = '0';
        item.style.transform = 'translateX(-20px)';
        
        setTimeout(() => {
            item.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
            item.style.opacity = '1';
            item.style.transform = 'translateX(0)';
        }, index * 50);
    });
}

/**
 * Initialize form validation
 */
function initializeFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.prototype.slice.call(forms).forEach(function(form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
                
                // Show first invalid field
                const firstInvalid = form.querySelector(':invalid');
                if (firstInvalid) {
                    firstInvalid.focus();
                    showToast('Please fill in all required fields', 'warning');
                }
            }
            
            form.classList.add('was-validated');
        }, false);
    });
}

/**
 * Initialize charts (if Chart.js is available)
 */
function initializeCharts() {
    if (typeof Chart === 'undefined') return;
    
    // Set default chart options
    Chart.defaults.font.family = 'Segoe UI, sans-serif';
    Chart.defaults.font.size = 12;
    Chart.defaults.color = '#6c757d';
    
    // Initialize specific charts based on page
    initializePerformanceChart();
    initializeComparisonChart();
}

/**
 * Initialize file upload handlers
 */
function initializeFileUploads() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(input => {
        input.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                validateFileUpload(file, input);
            }
        });
        
        // Add drag and drop support
        const parent = input.closest('.form-group') || input.parentElement;
        if (parent) {
            addDragDropSupport(parent, input);
        }
    });
}

/**
 * Validate file uploads
 */
function validateFileUpload(file, input) {
    const maxSize = 10 * 1024 * 1024; // 10MB
    const allowedTypes = ['text/csv', 'application/json', '.csv'];
    
    if (file.size > maxSize) {
        showToast('File too large. Maximum size is 10MB.', 'error');
        input.value = '';
        return false;
    }
    
    const fileName = file.name.toLowerCase();
    const isValidType = allowedTypes.some(type => 
        fileName.endsWith(type) || file.type === type
    );
    
    if (!isValidType) {
        showToast('Invalid file type. Please upload CSV or JSON files.', 'error');
        input.value = '';
        return false;
    }
    
    return true;
}

/**
 * Add drag and drop support to file inputs
 */
function addDragDropSupport(element, input) {
    element.addEventListener('dragover', function(e) {
        e.preventDefault();
        element.classList.add('drag-over');
    });
    
    element.addEventListener('dragleave', function(e) {
        e.preventDefault();
        element.classList.remove('drag-over');
    });
    
    element.addEventListener('drop', function(e) {
        e.preventDefault();
        element.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            input.files = files;
            const event = new Event('change', { bubbles: true });
            input.dispatchEvent(event);
        }
    });
}

/**
 * Show toast notifications
 */
function showToast(message, type = 'info') {
    const toastHtml = `
        <div class="toast align-items-center text-white bg-${type === 'error' ? 'danger' : type} border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-${getToastIcon(type)} me-2"></i>
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    // Create toast container if it doesn't exist
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        document.body.appendChild(container);
    }
    
    // Add toast to container
    container.insertAdjacentHTML('beforeend', toastHtml);
    
    // Show toast
    const toastElement = container.lastElementChild;
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
    
    // Remove toast element after it's hidden
    toastElement.addEventListener('hidden.bs.toast', function() {
        toastElement.remove();
    });
}

/**
 * Get appropriate icon for toast type
 */
function getToastIcon(type) {
    const icons = {
        success: 'check-circle',
        error: 'exclamation-circle',
        warning: 'exclamation-triangle',
        info: 'info-circle'
    };
    return icons[type] || 'info-circle';
}

/**
 * Initialize performance chart (dashboard)
 */
function initializePerformanceChart() {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Performance Score',
                data: [],
                borderColor: '#0d6efd',
                backgroundColor: 'rgba(13, 110, 253, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

/**
 * Initialize comparison chart
 */
function initializeComparisonChart() {
    const ctx = document.getElementById('comparisonChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Training Time (s)',
                data: [],
                backgroundColor: 'rgba(13, 110, 253, 0.8)'
            }, {
                label: 'Performance Score',
                data: [],
                backgroundColor: 'rgba(25, 135, 84, 0.8)'
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

/**
 * Loading state management
 */
function showLoading(element) {
    if (typeof element === 'string') {
        element = document.getElementById(element);
    }
    
    if (element) {
        element.classList.add('loading');
        const loadingSpinner = document.createElement('div');
        loadingSpinner.className = 'spinner-border spinner-border-sm me-2';
        loadingSpinner.setAttribute('role', 'status');
        element.insertBefore(loadingSpinner, element.firstChild);
    }
}

function hideLoading(element) {
    if (typeof element === 'string') {
        element = document.getElementById(element);
    }
    
    if (element) {
        element.classList.remove('loading');
        const spinner = element.querySelector('.spinner-border');
        if (spinner) {
            spinner.remove();
        }
    }
}

/**
 * AJAX helper functions
 */
function makeRequest(url, options = {}) {
    const defaultOptions = {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
        credentials: 'same-origin'
    };
    
    const config = { ...defaultOptions, ...options };
    
    return fetch(url, config)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .catch(error => {
            console.error('Request failed:', error);
            showToast('Request failed. Please try again.', 'error');
            throw error;
        });
}

/**
 * Format utilities
 */
function formatTime(seconds) {
    if (seconds < 60) {
        return `${seconds.toFixed(2)}s`;
    } else if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    }
}

function formatNumber(number, decimals = 2) {
    if (typeof number !== 'number' || isNaN(number)) {
        return 'N/A';
    }
    
    if (Math.abs(number) >= 1e9) {
        return (number / 1e9).toFixed(decimals) + 'B';
    } else if (Math.abs(number) >= 1e6) {
        return (number / 1e6).toFixed(decimals) + 'M';
    } else if (Math.abs(number) >= 1e3) {
        return (number / 1e3).toFixed(decimals) + 'K';
    } else {
        return number.toFixed(decimals);
    }
}

/**
 * URL utilities
 */
function updateURLParams(params) {
    const url = new URL(window.location);
    Object.keys(params).forEach(key => {
        if (params[key] !== null && params[key] !== undefined) {
            url.searchParams.set(key, params[key]);
        } else {
            url.searchParams.delete(key);
        }
    });
    window.history.replaceState({}, '', url);
}

function getURLParam(param, defaultValue = null) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(param) || defaultValue;
}

/**
 * Theme management (if dark mode is implemented)
 */
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const theme = savedTheme || (prefersDark ? 'dark' : 'light');
    
    setTheme(theme);
    
    // Listen for system theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addListener(e => {
        if (!localStorage.getItem('theme')) {
            setTheme(e.matches ? 'dark' : 'light');
        }
    });
}

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    
    // Update theme toggle button if exists
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.innerHTML = theme === 'dark' 
            ? '<i class="fas fa-sun"></i>' 
            : '<i class="fas fa-moon"></i>';
    }
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
}

/**
 * Export functions for global use
 */
window.MasterOptimization = {
    showToast,
    showLoading,
    hideLoading,
    makeRequest,
    formatTime,
    formatNumber,
    updateURLParams,
    getURLParam,
    toggleTheme
};