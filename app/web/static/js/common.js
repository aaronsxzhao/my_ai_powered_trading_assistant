/**
 * Common JavaScript utilities for AI Trading Coach
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


// ==================== SMOOTH NAVIGATION ====================

/**
 * Navigate to a URL with smooth transition
 * @param {string} url - URL to navigate to
 */
function navigateTo(url) {
    const transition = document.getElementById('page-transition');
    const loadingBar = document.getElementById('nav-loading-bar');
    
    if (loadingBar) loadingBar.classList.add('loading');
    if (transition) transition.classList.add('active');
    
    setTimeout(() => {
        window.location.href = url;
    }, 100);
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
        // Also close user menu
        const userMenu = document.getElementById('user-menu-dropdown');
        if (userMenu) userMenu.classList.add('hidden');
    }
});

// ==================== USER MENU ====================

/**
 * Toggle the user menu dropdown
 */
function toggleUserMenu(event) {
    event.stopPropagation();
    const dropdown = document.getElementById('user-menu-dropdown');
    if (dropdown) {
        dropdown.classList.toggle('hidden');
    }
}

// Close user menu when clicking outside
document.addEventListener('click', (e) => {
    const container = document.getElementById('user-menu-container');
    const dropdown = document.getElementById('user-menu-dropdown');
    if (container && dropdown && !container.contains(e.target)) {
        dropdown.classList.add('hidden');
    }
});


// ==================== KEYBOARD SHORTCUTS ====================

/**
 * Global keyboard shortcuts for power users
 * 
 * Shortcuts:
 * - Ctrl/Cmd + N: New trade (go to /add-trade)
 * - Ctrl/Cmd + E: Export trades (on trades page)
 * - Ctrl/Cmd + /: Show shortcuts help
 * - G then T: Go to Trades
 * - G then D: Go to Dashboard
 * - G then S: Go to Stats
 * - G then A: Go to Add Trade
 * - J/K: Navigate trades (on trades page)
 */

let pendingGoKey = false;
let goKeyTimeout = null;

document.addEventListener('keydown', (e) => {
    // Skip if user is typing in an input/textarea
    const tag = e.target.tagName.toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select' || e.target.isContentEditable) {
        return;
    }
    
    const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
    const modKey = isMac ? e.metaKey : e.ctrlKey;
    
    // Ctrl/Cmd + N: New trade
    if (modKey && e.key === 'n') {
        e.preventDefault();
        navigateTo('/add-trade');
        return;
    }
    
    // Ctrl/Cmd + E: Export (on trades page)
    if (modKey && e.key === 'e') {
        if (typeof exportTrades === 'function') {
            e.preventDefault();
            exportTrades();
            return;
        }
    }
    
    // Ctrl/Cmd + /: Show shortcuts help
    if (modKey && e.key === '/') {
        e.preventDefault();
        showShortcutsHelp();
        return;
    }
    
    // G-prefix shortcuts (Gmail-style)
    if (e.key === 'g' && !modKey) {
        pendingGoKey = true;
        clearTimeout(goKeyTimeout);
        goKeyTimeout = setTimeout(() => { pendingGoKey = false; }, 1000);
        return;
    }
    
    if (pendingGoKey) {
        pendingGoKey = false;
        clearTimeout(goKeyTimeout);
        
        switch (e.key.toLowerCase()) {
            case 'd': // Go to Dashboard
                navigateTo('/');
                break;
            case 't': // Go to Trades
                navigateTo('/trades');
                break;
            case 's': // Go to Stats
                navigateTo('/stats');
                break;
            case 'a': // Go to Add Trade
                navigateTo('/add-trade');
                break;
        }
        return;
    }
    
    // ? key: Show shortcuts help
    if (e.key === '?' && e.shiftKey) {
        e.preventDefault();
        showShortcutsHelp();
        return;
    }
});

/**
 * Show keyboard shortcuts help modal
 */
function showShortcutsHelp() {
    // Remove existing help if present
    const existing = document.getElementById('shortcuts-help-modal');
    if (existing) {
        existing.remove();
        return;
    }
    
    const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
    const modKeyName = isMac ? 'Cmd' : 'Ctrl';
    
    const modal = document.createElement('div');
    modal.id = 'shortcuts-help-modal';
    modal.className = 'fixed inset-0 bg-dark-950/80 backdrop-blur-sm flex items-center justify-center z-50';
    modal.onclick = (e) => { if (e.target === modal) modal.remove(); };
    
    modal.innerHTML = `
        <div class="bg-dark-800 rounded-2xl p-6 max-w-md w-full mx-4 border border-dark-600 shadow-2xl">
            <div class="flex justify-between items-center mb-4">
                <h2 class="text-xl font-bold text-white">Keyboard Shortcuts</h2>
                <button onclick="this.closest('#shortcuts-help-modal').remove()" class="text-dark-400 hover:text-white">&times;</button>
            </div>
            <div class="space-y-4 text-sm">
                <div>
                    <h3 class="font-semibold text-accent-400 mb-2">Navigation</h3>
                    <div class="space-y-1 text-dark-300">
                        <div class="flex justify-between"><span>Go to Dashboard</span><kbd class="kbd">g d</kbd></div>
                        <div class="flex justify-between"><span>Go to Trades</span><kbd class="kbd">g t</kbd></div>
                        <div class="flex justify-between"><span>Go to Stats</span><kbd class="kbd">g s</kbd></div>
                        <div class="flex justify-between"><span>Go to Add Trade</span><kbd class="kbd">g a</kbd></div>
                    </div>
                </div>
                <div>
                    <h3 class="font-semibold text-accent-400 mb-2">Actions</h3>
                    <div class="space-y-1 text-dark-300">
                        <div class="flex justify-between"><span>New Trade</span><kbd class="kbd">${modKeyName}+N</kbd></div>
                        <div class="flex justify-between"><span>Export Trades</span><kbd class="kbd">${modKeyName}+E</kbd></div>
                        <div class="flex justify-between"><span>Show Shortcuts</span><kbd class="kbd">${modKeyName}+/</kbd></div>
                    </div>
                </div>
                <div>
                    <h3 class="font-semibold text-accent-400 mb-2">General</h3>
                    <div class="space-y-1 text-dark-300">
                        <div class="flex justify-between"><span>Close Modal</span><kbd class="kbd">Esc</kbd></div>
                    </div>
                </div>
            </div>
            <p class="text-dark-500 text-xs mt-4">Press <kbd class="kbd text-xs">?</kbd> anytime to show this help</p>
        </div>
    `;
    
    document.body.appendChild(modal);
}

// Add kbd styling
const style = document.createElement('style');
style.textContent = `
    .kbd {
        display: inline-block;
        padding: 2px 6px;
        font-family: monospace;
        font-size: 11px;
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 4px;
        color: #94a3b8;
    }
`;
document.head.appendChild(style);
