/**
 * Trades page JavaScript
 * 
 * Contains functionality for the trade journal page:
 * - Trade editing modal
 * - Trade deletion
 * - Bulk operations (recalculate, analyze all)
 * - Currency rate fetching
 */

// ==================== STATE ====================

let currentTradeId = null;
let currentTradeDate = null;


// ==================== MODAL FUNCTIONS ====================

/**
 * Open the edit trade modal
 */
function openEditModal(id, ticker, shares, entry, exit, sl, tp, currency, rate, entryTime, exitTime, timeframe, direction) {
    currentTradeId = id;
    currentTradeDate = exitTime || entryTime;
    
    document.getElementById('edit-trade-id').textContent = id;
    document.getElementById('edit-id').value = id;
    document.getElementById('edit-ticker').value = ticker || '';
    document.getElementById('edit-direction').value = direction || 'long';
    document.getElementById('edit-shares').value = shares;
    document.getElementById('edit-entry').value = entry;
    document.getElementById('edit-exit').value = exit;
    document.getElementById('edit-sl').value = sl || '';
    document.getElementById('edit-tp').value = tp || '';
    document.getElementById('edit-currency').value = currency || 'USD';
    document.getElementById('edit-rate').value = rate;
    document.getElementById('edit-timeframe').value = timeframe || '5m';

    if (entryTime) document.getElementById('edit-entry-time').value = entryTime;
    if (exitTime) document.getElementById('edit-exit-time').value = exitTime;
    
    document.getElementById('edit-date-display').textContent = 
        currentTradeDate ? `Trade date: ${currentTradeDate.split('T')[0]}` : '';
    
    document.getElementById('edit-modal').classList.remove('hidden');
}

/**
 * Close the edit trade modal
 */
function closeEditModal() {
    document.getElementById('edit-modal').classList.add('hidden');
    currentTradeId = null;
}


// ==================== TRADE OPERATIONS ====================

/**
 * Fetch historical exchange rate for a currency
 */
async function fetchHistoricalRate() {
    const currency = document.getElementById('edit-currency').value;
    if (currency === 'USD') {
        document.getElementById('edit-rate').value = 1.0;
        return;
    }
    
    const date = currentTradeDate ? currentTradeDate.split('T')[0] : new Date().toISOString().split('T')[0];
    
    try {
        const response = await fetch(`/api/exchange-rate?currency=${currency}&date=${date}`);
        const data = await response.json();
        
        if (data.rate) {
            document.getElementById('edit-rate').value = data.rate;
            showToast(`Rate fetched: 1 USD = ${data.rate} ${currency}`);
        } else {
            showToast('Could not fetch rate', 'error');
        }
    } catch (err) {
        showToast('Error fetching rate', 'error');
    }
}

/**
 * Delete a single trade
 */
async function deleteTrade(id) {
    if (!confirm('Delete this trade?')) return;
    
    try {
        const response = await fetch(`/api/trades/${id}`, { method: 'DELETE' });
        if (response.ok) {
            showToast('Trade deleted');
            setTimeout(() => window.location.reload(), 500);
        }
    } catch (err) {
        showToast('Error deleting trade', 'error');
    }
}


// ==================== BULK OPERATIONS ====================

/**
 * Recalculate metrics for all trades
 */
async function recalculateMetrics() {
    showStartToast('Recalculating metrics');
    try {
        const response = await fetch('/api/recalculate-metrics', { method: 'POST' });
        const data = await response.json();
        showFinishToast('Metrics recalculated', `${data.count || 'all'} trades updated`);
        setTimeout(() => window.location.reload(), 1500);
    } catch (err) {
        showToast('Recalculation failed', 'error');
    }
}

/**
 * Analyze all trades with AI
 */
async function analyzeAllTrades() {
    // Ask if force re-analyze
    const force = confirm('Analyze ALL trades? (Cancel to only analyze unreviewed trades)');

    // Show persistent loading message
    showToast('🧠 AI analysis started... This may take 1-2 minutes for many trades.', 'info', 120000);
    console.log('Analyze started, force=' + force);
    
    try {
        const url = force ? '/api/analyze-all-trades?force=true' : '/api/analyze-all-trades';
        console.log('Calling: ' + url);
        
        // Use AbortController for timeout (10 minutes)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 600000);
        
        const response = await fetch(url, { 
            method: 'POST',
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        
        console.log('Response status: ' + response.status);
        const data = await response.json();
        console.log('Response data:', data);

        if (data.error) {
            showToast(data.error, 'error');
            return;
        }

        const analyzed = data.analyzed || 0;
        const skipped = data.skipped || 0;
        const errors = data.errors || 0;

        let msg = `${analyzed} analyzed`;
        if (skipped > 0) msg += `, ${skipped} skipped (already done)`;
        if (errors > 0) msg += `, ${errors} failed`;

        showToast('✅ ' + msg, 'success', 5000);
        setTimeout(() => window.location.reload(), 2000);
    } catch (err) {
        console.error('Analyze error:', err);
        if (err.name === 'AbortError') {
            showToast('Analysis timed out. Try again or analyze individual trades.', 'error');
        } else {
            showToast('Analysis failed: ' + err.message, 'error');
        }
    }
}

/**
 * Delete all trades (with double confirmation)
 */
async function deleteAllTrades() {
    if (!confirm('⚠️ Delete ALL trades? This cannot be undone!')) return;
    if (!confirm('Are you REALLY sure? This will delete your entire trade history!')) return;
    
    showStartToast('Deleting all trades');
    try {
        await fetch('/api/trades', { method: 'DELETE' });
        showFinishToast('All trades deleted');
        setTimeout(() => window.location.reload(), 1000);
    } catch (err) {
        showToast('Delete failed', 'error');
    }
}


// ==================== INITIALIZATION ====================

document.addEventListener('DOMContentLoaded', function() {
    // Auto-fetch rate when currency changes
    const currencySelect = document.getElementById('edit-currency');
    if (currencySelect) {
        currencySelect.addEventListener('change', fetchHistoricalRate);
    }
    
    // Handle edit form submission
    const editForm = document.getElementById('edit-form');
    if (editForm) {
        editForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const id = document.getElementById('edit-id').value;
            const slValue = document.getElementById('edit-sl').value;
            const tpValue = document.getElementById('edit-tp').value;
            
            const data = {
                ticker: document.getElementById('edit-ticker').value.toUpperCase(),
                direction: document.getElementById('edit-direction').value,
                size: parseFloat(document.getElementById('edit-shares').value),
                entry_price: parseFloat(document.getElementById('edit-entry').value),
                exit_price: parseFloat(document.getElementById('edit-exit').value),
                stop_loss: slValue ? parseFloat(slValue) : null,
                take_profit: tpValue ? parseFloat(tpValue) : null,
                currency: document.getElementById('edit-currency').value,
                currency_rate: parseFloat(document.getElementById('edit-rate').value),
                entry_time: document.getElementById('edit-entry-time').value || null,
                exit_time: document.getElementById('edit-exit-time').value || null,
                timeframe: document.getElementById('edit-timeframe').value,
            };
            
            try {
                const response = await fetch(`/api/trades/${id}`, {
                    method: 'PATCH',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                if (response.ok) {
                    showToast('Trade updated!');
                    closeEditModal();
                    setTimeout(() => window.location.reload(), 1000);
                } else {
                    showToast('Failed to update trade', 'error');
                }
            } catch (err) {
                showToast('Error: ' + err.message, 'error');
            }
        });
    }
});
