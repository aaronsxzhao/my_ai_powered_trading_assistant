/**
 * Common JavaScript utilities for Brooks Trading Coach
 * 
 * Contains shared functionality used across multiple pages:
 * - Toast notifications
 * - Loading states
 * - Theme management
 * - API helpers
 */

// ==================== TOAST NOTIFICATIONS ====================

/**
 * Show a toast notification
 * @param {string} message - Message to display
 * @param {string} type - Type: 'success', 'error', 'info', 'start', 'finish', 'loading'
 * @param {number} duration - Duration in ms (0 = persistent)
 */
function showToast(message, type = 'success', duration = 3000) {
    const toast = document.getElementById('toast');
    const icon = document.getElementById('toast-icon');
    const msg = document.getElementById('toast-message');
    
    if (!toast || !icon || !msg) return;
    
    const icons = {
        'success': '✓',
        'error': '✕',
        'info': 'ℹ',
        'start': '🚀',
        'finish': '✅',
        'loading': '⏳'
    };
    icon.textContent = icons[type] || '✓';
    msg.textContent = message;
    
    toast.classList.remove('hidden');
    if (duration > 0) {
        setTimeout(() => toast.classList.add('hidden'), duration);
    }
}

/**
 * Show a quick "starting" notification
 * @param {string} action - Action name
 */
function showStartToast(action) {
    showToast(`Starting ${action}...`, 'start', 500);
}

/**
 * Show a "finished" notification
 * @param {string} action - Action name
 * @param {string} brief - Optional brief result message
 */
function showFinishToast(action, brief = '') {
    const msg = brief ? `${action} complete: ${brief}` : `${action} complete!`;
    showToast(msg, 'finish', 3000);
}


// ==================== LOADING STATES ====================

/**
 * Show/hide the full-page loading overlay (deprecated - prefer local loading states)
 * @param {boolean} show - Whether to show the loading overlay
 */
function showLoading(show) {
    const loading = document.getElementById('loading');
    if (loading) {
        loading.classList.toggle('hidden', !show);
    }
}


// ==================== THEME MANAGEMENT ====================

/**
 * Toggle between light and dark themes
 */
function toggleTheme() {
    const html = document.documentElement;
    const isDark = html.classList.contains('dark');
    
    if (isDark) {
        html.classList.remove('dark');
        localStorage.setItem('theme', 'light');
        updateThemeIcon('☀️');
    } else {
        html.classList.add('dark');
        localStorage.setItem('theme', 'dark');
        updateThemeIcon('🌙');
    }
}

/**
 * Update the theme icon
 * @param {string} icon - Icon character
 */
function updateThemeIcon(icon) {
    const themeIcon = document.getElementById('theme-icon');
    if (themeIcon) {
        themeIcon.textContent = icon;
    }
}

/**
 * Initialize theme from saved preference or system preference
 */
function initTheme() {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const useDark = savedTheme === 'dark' || (!savedTheme && prefersDark);
    
    if (useDark) {
        document.documentElement.classList.add('dark');
        updateThemeIcon('🌙');
    } else {
        document.documentElement.classList.remove('dark');
        updateThemeIcon('☀️');
    }
}

// Initialize theme on load
initTheme();


// ==================== API HELPERS ====================

/**
 * Make an API request with standard error handling
 * @param {string} url - API endpoint URL
 * @param {object} options - Fetch options
 * @returns {Promise<object>} - Response data
 */
async function apiRequest(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || data.detail || `HTTP ${response.status}`);
        }
        
        return data;
    } catch (error) {
        console.error(`API Error (${url}):`, error);
        throw error;
    }
}

/**
 * Make a GET request
 * @param {string} url - API endpoint
 * @returns {Promise<object>}
 */
async function apiGet(url) {
    return apiRequest(url, { method: 'GET' });
}

/**
 * Make a POST request
 * @param {string} url - API endpoint
 * @param {object} data - Request body
 * @returns {Promise<object>}
 */
async function apiPost(url, data) {
    return apiRequest(url, {
        method: 'POST',
        body: JSON.stringify(data),
    });
}

/**
 * Make a PATCH request
 * @param {string} url - API endpoint
 * @param {object} data - Request body
 * @returns {Promise<object>}
 */
async function apiPatch(url, data) {
    return apiRequest(url, {
        method: 'PATCH',
        body: JSON.stringify(data),
    });
}

/**
 * Make a DELETE request
 * @param {string} url - API endpoint
 * @returns {Promise<object>}
 */
async function apiDelete(url) {
    return apiRequest(url, { method: 'DELETE' });
}


// ==================== UTILITY FUNCTIONS ====================

/**
 * Format a number as currency
 * @param {number} value - Value to format
 * @param {string} currency - Currency code (default: USD)
 * @returns {string}
 */
function formatCurrency(value, currency = 'USD') {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: currency,
    }).format(value);
}

/**
 * Format a number with specified decimal places
 * @param {number} value - Value to format
 * @param {number} decimals - Number of decimal places
 * @returns {string}
 */
function formatNumber(value, decimals = 2) {
    return Number(value).toFixed(decimals);
}

/**
 * Debounce a function
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in ms
 * @returns {Function}
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Show a confirmation dialog
 * @param {string} message - Confirmation message
 * @returns {boolean}
 */
function confirmAction(message) {
    return window.confirm(message);
}


// ==================== MODAL HELPERS ====================

/**
 * Open a modal by ID
 * @param {string} modalId - Modal element ID
 */
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('hidden');
    }
}

/**
 * Close a modal by ID
 * @param {string} modalId - Modal element ID
 */
function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('hidden');
    }
}

// Close modals on escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.querySelectorAll('[id$="-modal"]:not(.hidden)').forEach(modal => {
            modal.classList.add('hidden');
        });
    }
});
