// ==========================================
// MORPH AI - ENHANCED INTERACTIONS & ANIMATIONS
// Professional UI/UX JavaScript Enhancements
// ==========================================

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initializeAlerts();
    initializeAnimations();
    initializeNavigation();
    initializeFormEnhancements();
    initializeImageLazyLoading();
    initializeTooltips();
    initializeScrollEffects();
    initializeKeyboardShortcuts();
    initializePerformanceMonitoring();
    initializeScrollAnimations();
    initializeHeroParallax();
    
    // Initialize accessibility enhancements
    enhanceFormValidation();
    createAnnouncementRegion();
    
    // Set initial ARIA states
    document.querySelectorAll('.morph-view-toggle button').forEach((btn, index) => {
        btn.setAttribute('aria-pressed', index === 0 ? 'true' : 'false');
    });
    
    // Make main content focusable for skip link
    const mainContent = document.getElementById('main-content');
    if (mainContent) {
        mainContent.setAttribute('tabindex', '-1');
    }
});

// ==========================================
// ALERT MANAGEMENT
// ==========================================
function initializeAlerts() {
    const alerts = document.querySelectorAll('.alert');
    
    alerts.forEach(function(alert) {
        // Add slide-in animation
        alert.classList.add('fade-in-up');
        
        // Auto-hide after 5 seconds
        setTimeout(function() {
            if (alert.parentNode) {
                alert.style.opacity = '0';
                alert.style.transform = 'translateY(-20px)';
                setTimeout(() => {
                    if (alert.parentNode) {
                        alert.remove();
                    }
                }, 300);
            }
        }, 5000);
        
        // Enhanced close button
        const closeBtn = alert.querySelector('.btn-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', function() {
                alert.style.opacity = '0';
                alert.style.transform = 'translateY(-20px)';
                setTimeout(() => alert.remove(), 300);
            });
        }
    });
}

// ==========================================
// SMOOTH ANIMATIONS
// ==========================================
function initializeAnimations() {
    // Intersection Observer for scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe elements for scroll animations
    const animateElements = document.querySelectorAll('.card, .feature-card, .stat-card, .image-card');
    animateElements.forEach(el => {
        observer.observe(el);
    });
    
    // Staggered animations for grid items
    const gridItems = document.querySelectorAll('.row .col-md-4, .row .col-lg-3, .row .col-xl-3');
    gridItems.forEach((item, index) => {
        item.style.animationDelay = `${index * 0.1}s`;
    });
    
    // Parallax effect for hero section
    const heroSection = document.querySelector('.hero-section');
    if (heroSection) {
        window.addEventListener('scroll', function() {
            const scrolled = window.pageYOffset;
            const rate = scrolled * -0.5;
            heroSection.style.transform = `translate3d(0, ${rate}px, 0)`;
        });
    }
}

// ==========================================
// NAVIGATION ENHANCEMENTS
// ==========================================
function initializeNavigation() {
    const navbar = document.querySelector('.navbar');
    
    // Navbar scroll behavior
    let lastScrollTop = 0;
    window.addEventListener('scroll', function() {
        const currentScrollTop = window.pageYOffset || document.documentElement.scrollTop;
        
        if (currentScrollTop > lastScrollTop && currentScrollTop > 100) {
            // Scrolling down
            navbar.style.transform = 'translateY(-100%)';
        } else {
            // Scrolling up
            navbar.style.transform = 'translateY(0)';
        }
        
        // Add background on scroll
        if (currentScrollTop > 50) {
            navbar.classList.add('scrolled');
            navbar.style.backgroundColor = 'rgba(255, 255, 255, 0.95)';
            navbar.style.backdropFilter = 'blur(10px)';
        } else {
            navbar.classList.remove('scrolled');
            navbar.style.backgroundColor = '';
            navbar.style.backdropFilter = '';
        }
        
        lastScrollTop = currentScrollTop <= 0 ? 0 : currentScrollTop;
    });
    
    // Active navigation highlighting
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    const currentPath = window.location.pathname;
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
        
        // Hover effects
        link.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
        });
        
        link.addEventListener('mouseleave', function() {
            this.style.transform = '';
        });
    });
    
    // Mobile menu enhancements
    const navbarToggler = document.querySelector('.navbar-toggler');
    const navbarCollapse = document.querySelector('.navbar-collapse');
    
    if (navbarToggler && navbarCollapse) {
        navbarToggler.addEventListener('click', function() {
            setTimeout(() => {
                if (navbarCollapse.classList.contains('show')) {
                    navbarCollapse.style.animation = 'slideDown 0.3s ease-out';
                } else {
                    navbarCollapse.style.animation = 'slideUp 0.3s ease-out';
                }
            }, 10);
        });
    }
}

// ==========================================
// FORM ENHANCEMENTS
// ==========================================
function initializeFormEnhancements() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(function(form) {
        // Enhanced validation
        form.addEventListener('submit', function(e) {
            if (!form.checkValidity()) {
                e.preventDefault();
                e.stopPropagation();
                
                // Smooth scroll to first invalid field
                const firstInvalid = form.querySelector(':invalid');
                if (firstInvalid) {
                    firstInvalid.scrollIntoView({
                        behavior: 'smooth',
                        block: 'center'
                    });
                    firstInvalid.focus();
                }
            } else {
                // Show loading state
                const submitBtn = form.querySelector('button[type="submit"], input[type="submit"]');
                if (submitBtn) {
                    const originalText = submitBtn.innerHTML;
                    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
                    submitBtn.disabled = true;
                    
                    // Restore button after 5 seconds if form doesn't redirect
                    setTimeout(() => {
                        if (submitBtn.disabled) {
                            submitBtn.innerHTML = originalText;
                            submitBtn.disabled = false;
                        }
                    }, 5000);
                }
            }
            
            form.classList.add('was-validated');
        });
        
        // Real-time validation feedback
        const inputs = form.querySelectorAll('input, textarea, select');
        inputs.forEach(input => {
            input.addEventListener('blur', function() {
                if (this.checkValidity()) {
                    this.classList.remove('is-invalid');
                    this.classList.add('is-valid');
                } else {
                    this.classList.remove('is-valid');
                    this.classList.add('is-invalid');
                }
            });
            
            // Enhanced focus effects
            input.addEventListener('focus', function() {
                this.style.transform = 'scale(1.02)';
                this.style.transition = 'all 0.2s ease';
            });
            
            input.addEventListener('blur', function() {
                this.style.transform = '';
            });
        });
    });
    
    // File input enhancements
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            const files = this.files;
            if (files.length > 0) {
                // Create file preview
                createFilePreview(files[0], this);
            }
        });
    });
}

// ==========================================
// IMAGE LAZY LOADING
// ==========================================
function initializeImageLazyLoading() {
    const images = document.querySelectorAll('img[loading="lazy"]');
    
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    
                    // Add loading animation
                    img.style.opacity = '0';
                    img.style.transform = 'scale(0.8)';
                    
                    img.addEventListener('load', function() {
                        img.style.opacity = '1';
                        img.style.transform = 'scale(1)';
                        img.style.transition = 'all 0.3s ease';
                    });
                    
                    imageObserver.unobserve(img);
                }
            });
        });
        
        images.forEach(img => imageObserver.observe(img));
    }
    
    // Image zoom on click
    const zoomableImages = document.querySelectorAll('.image-card img, .card-img-top');
    zoomableImages.forEach(img => {
        img.addEventListener('click', function() {
            createImageModal(this.src, this.alt);
        });
        
        img.style.cursor = 'zoom-in';
    });
}

// ==========================================
// TOOLTIP INITIALIZATION
// ==========================================
function initializeTooltips() {
    // Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"], [title]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Custom tooltips for buttons
    const buttons = document.querySelectorAll('.btn-icon');
    buttons.forEach(btn => {
        btn.addEventListener('mouseenter', function() {
            const title = this.getAttribute('title');
            if (title) {
                showCustomTooltip(this, title);
            }
        });
        
        btn.addEventListener('mouseleave', function() {
            hideCustomTooltip();
        });
    });
}

// ==========================================
// SCROLL EFFECTS
// ==========================================
function initializeScrollEffects() {
    // Smooth scrolling for anchor links
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(function(link) {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Scroll to top button
    createScrollToTopButton();
    
    // Progress bar on scroll
    createScrollProgressBar();
}

// ==========================================
// ACCESSIBILITY & KEYBOARD SHORTCUTS
// ==========================================
function initializeKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Upload shortcut (Ctrl/Cmd + U)
        if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
            e.preventDefault();
            const uploadLink = document.querySelector('a[href*="upload"]');
            if (uploadLink) uploadLink.click();
        }
        
        // Search shortcut (Ctrl/Cmd + K)
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.querySelector('#search-input, input[type="search"]');
            if (searchInput) {
                searchInput.focus();
                searchInput.select();
            }
        }
        
        // Gallery view toggle (G)
        if (e.key === 'g' && !e.ctrlKey && !e.metaKey) {
            const viewToggle = document.querySelector('.morph-view-toggle button:not([aria-pressed="true"])');
            if (viewToggle && document.activeElement.tagName !== 'INPUT') {
                viewToggle.click();
            }
        }
        
        // Skip to main content (Alt + M)
        if (e.altKey && e.key === 'm') {
            e.preventDefault();
            const mainContent = document.getElementById('main-content');
            if (mainContent) {
                mainContent.focus();
                mainContent.scrollIntoView({ behavior: 'smooth' });
            }
        }
        
        // Escape key handling
        if (e.key === 'Escape') {
            // Close modals, dropdowns, etc.
            const modal = document.querySelector('.modal.show');
            if (modal) {
                bootstrap.Modal.getInstance(modal).hide();
            }
            
            const dropdowns = document.querySelectorAll('.dropdown-menu.show');
            dropdowns.forEach(dropdown => {
                bootstrap.Dropdown.getInstance(dropdown.previousElementSibling).hide();
            });
        }
    });
    
    // Enhanced focus management for upload zone
    const uploadZone = document.getElementById('upload-zone');
    if (uploadZone) {
        uploadZone.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                const fileInput = document.getElementById('file-input');
                if (fileInput) fileInput.click();
            }
        });
    }
    
    // ARIA state updates for view toggle buttons
    const viewToggleButtons = document.querySelectorAll('.morph-view-toggle button');
    viewToggleButtons.forEach(button => {
        button.addEventListener('click', function() {
            viewToggleButtons.forEach(btn => btn.setAttribute('aria-pressed', 'false'));
            this.setAttribute('aria-pressed', 'true');
            
            // Announce view change to screen readers
            announceToScreenReader(`View changed to ${this.dataset.view} layout`);
        });
    });
}

// Screen reader announcements
function announceToScreenReader(message) {
    const announcement = document.getElementById('sr-announcements') || createAnnouncementRegion();
    announcement.textContent = message;
    
    // Clear after announcement
    setTimeout(() => {
        announcement.textContent = '';
    }, 1000);
}

function createAnnouncementRegion() {
    const region = document.createElement('div');
    region.id = 'sr-announcements';
    region.setAttribute('aria-live', 'polite');
    region.setAttribute('aria-atomic', 'true');
    region.className = 'sr-only';
    document.body.appendChild(region);
    return region;
}

// Enhanced form validation with ARIA feedback
function enhanceFormValidation() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        const inputs = form.querySelectorAll('input, textarea, select');
        
        inputs.forEach(input => {
            // Real-time validation with ARIA feedback
            input.addEventListener('invalid', function(e) {
                e.preventDefault();
                
                // Create or update error message
                const errorId = this.id + '-error';
                let errorMessage = document.getElementById(errorId);
                
                if (!errorMessage) {
                    errorMessage = document.createElement('div');
                    errorMessage.id = errorId;
                    errorMessage.className = 'invalid-feedback d-block';
                    errorMessage.setAttribute('role', 'alert');
                    this.parentNode.appendChild(errorMessage);
                }
                
                // Set error message and ARIA attributes
                errorMessage.textContent = this.validationMessage;
                this.setAttribute('aria-describedby', errorId);
                this.setAttribute('aria-invalid', 'true');
                
                // Visual feedback
                this.classList.add('is-invalid');
                
                // Announce error to screen readers
                announceToScreenReader(`Error: ${this.validationMessage}`);
            });
            
            input.addEventListener('input', function() {
                if (this.checkValidity()) {
                    // Clear error state
                    this.classList.remove('is-invalid');
                    this.classList.add('is-valid');
                    this.setAttribute('aria-invalid', 'false');
                    
                    const errorId = this.id + '-error';
                    const errorMessage = document.getElementById(errorId);
                    if (errorMessage) {
                        errorMessage.remove();
                        this.removeAttribute('aria-describedby');
                    }
                }
            });
        });
    });
}

// ==========================================
// PERFORMANCE MONITORING
// ==========================================
function initializePerformanceMonitoring() {
    // Monitor page load performance
    window.addEventListener('load', function() {
        setTimeout(() => {
            const perfData = performance.getEntriesByType('navigation')[0];
            if (perfData) {
                console.log('Page Load Performance:', {
                    'Load Time': Math.round(perfData.loadEventEnd - perfData.loadEventStart) + 'ms',
                    'DOM Ready': Math.round(perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart) + 'ms',
                    'Total Time': Math.round(perfData.loadEventEnd - perfData.fetchStart) + 'ms'
                });
            }
        }, 0);
    });
    
    // Monitor long tasks
    if ('PerformanceObserver' in window) {
        const observer = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                if (entry.duration > 50) {
                    console.warn('Long task detected:', entry.duration + 'ms');
                }
            }
        });
        
        observer.observe({entryTypes: ['longtask']});
    }
}

// ==========================================
// UTILITY FUNCTIONS
// ==========================================

// Create file preview
function createFilePreview(file, input) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const preview = document.createElement('div');
        preview.className = 'file-preview mt-2 p-2 border rounded';
        preview.innerHTML = `
            <div class="d-flex align-items-center">
                <img src="${e.target.result}" alt="Preview" style="width: 50px; height: 50px; object-fit: cover; border-radius: 4px;">
                <div class="ms-2">
                    <div class="fw-bold">${file.name}</div>
                    <small class="text-muted">${formatFileSize(file.size)}</small>
                </div>
                <button type="button" class="btn btn-sm btn-outline-danger ms-auto" onclick="this.parentElement.parentElement.remove(); document.querySelector('input[type=file]').value = '';">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        // Remove existing preview
        const existingPreview = input.parentNode.querySelector('.file-preview');
        if (existingPreview) existingPreview.remove();
        
        input.parentNode.appendChild(preview);
    };
    
    if (file.type.startsWith('image/')) {
        reader.readAsDataURL(file);
    }
}

// Create image modal
function createImageModal(src, alt) {
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.innerHTML = `
        <div class="modal-dialog modal-lg modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">${alt || 'Image Preview'}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body text-center">
                    <img src="${src}" alt="${alt}" class="img-fluid rounded">
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    const bsModal = new bootstrap.Modal(modal);
    
    modal.addEventListener('hidden.bs.modal', function() {
        modal.remove();
    });
    
    bsModal.show();
}

// Custom tooltip functions
function showCustomTooltip(element, text) {
    const tooltip = document.createElement('div');
    tooltip.className = 'custom-tooltip';
    tooltip.textContent = text;
    tooltip.style.cssText = `
        position: absolute;
        background: var(--dark-navy);
        color: white;
        padding: 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        z-index: 1060;
        opacity: 0;
        transform: translateY(10px);
        transition: all 0.2s ease;
        pointer-events: none;
    `;
    
    document.body.appendChild(tooltip);
    
    const rect = element.getBoundingClientRect();
    tooltip.style.left = (rect.left + rect.width / 2 - tooltip.offsetWidth / 2) + 'px';
    tooltip.style.top = (rect.bottom + 5) + 'px';
    
    setTimeout(() => {
        tooltip.style.opacity = '1';
        tooltip.style.transform = 'translateY(0)';
    }, 10);
    
    tooltip.id = 'custom-tooltip';
}

function hideCustomTooltip() {
    const tooltip = document.getElementById('custom-tooltip');
    if (tooltip) {
        tooltip.style.opacity = '0';
        tooltip.style.transform = 'translateY(10px)';
        setTimeout(() => tooltip.remove(), 200);
    }
}

// Scroll to top button
function createScrollToTopButton() {
    const button = document.createElement('button');
    button.innerHTML = '<i class="fas fa-arrow-up"></i>';
    button.className = 'btn btn-primary scroll-to-top';
    button.style.cssText = `
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: none;
        z-index: 1050;
        box-shadow: var(--shadow-lg);
        transition: all 0.3s ease;
    `;
    
    button.addEventListener('click', function() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
    
    document.body.appendChild(button);
    
    window.addEventListener('scroll', function() {
        if (window.pageYOffset > 300) {
            button.style.display = 'flex';
            button.style.alignItems = 'center';
            button.style.justifyContent = 'center';
        } else {
            button.style.display = 'none';
        }
    });
}

// Scroll progress bar
function createScrollProgressBar() {
    const progressBar = document.createElement('div');
    progressBar.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 0%;
        height: 3px;
        background: var(--primary-gradient);
        z-index: 1051;
        transition: width 0.1s ease;
    `;
    
    document.body.appendChild(progressBar);
    
    window.addEventListener('scroll', function() {
        const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
        const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const scrolled = (winScroll / height) * 100;
        progressBar.style.width = scrolled + '%';
    });
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Debounce function for performance
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

// Throttle function for scroll events
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// ==========================================
// ADDITIONAL CSS ANIMATIONS
// ==========================================
const additionalStyles = `
@keyframes slideDown {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideUp {
    from { opacity: 1; transform: translateY(0); }
    to { opacity: 0; transform: translateY(-10px); }
}

.scroll-to-top:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-xl);
}

.custom-tooltip {
    box-shadow: var(--shadow-md);
}

.navbar {
    transition: all 0.3s ease;
}

.btn:active {
    transform: translateY(1px);
}

.card:hover .btn {
    transform: translateY(-1px);
}
`;

// Inject additional styles
const styleSheet = document.createElement('style');
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);

// ==========================================
// SCROLL-BASED ANIMATIONS
// ==========================================
function initializeScrollAnimations() {
    // Enhanced scroll observer for animate-on-scroll elements
    const observerOptions = {
        threshold: 0.15,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const scrollObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate');
                // Add staggered delays for child elements
                const children = entry.target.querySelectorAll('.card, .btn, h1, h2, h3, p');
                children.forEach((child, index) => {
                    setTimeout(() => {
                        child.style.opacity = '1';
                        child.style.transform = 'translateY(0)';
                    }, index * 100);
                });
            }
        });
    }, observerOptions);
    
    // Observe elements with animate-on-scroll class
    const animateElements = document.querySelectorAll('.animate-on-scroll');
    animateElements.forEach(el => scrollObserver.observe(el));
    
    // Stats counter animation
    const statNumbers = document.querySelectorAll('.hero-stat-number, .stat-number');
    const statsObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateNumber(entry.target);
                statsObserver.unobserve(entry.target);
            }
        });
    });
    
    statNumbers.forEach(stat => statsObserver.observe(stat));
}

// ==========================================
// HERO PARALLAX EFFECT
// ==========================================
function initializeHeroParallax() {
    const heroSection = document.querySelector('.hero-section');
    const floatingCells = document.querySelectorAll('.floating-cell');
    
    if (!heroSection) return;
    
    // Enhanced parallax with different speeds for floating cells
    const handleScroll = throttle(() => {
        const scrolled = window.pageYOffset;
        const heroHeight = heroSection.offsetHeight;
        const scrollProgress = Math.min(scrolled / heroHeight, 1);
        
        // Parallax for hero content
        const heroContent = heroSection.querySelector('.hero-content');
        if (heroContent) {
            heroContent.style.transform = `translateY(${scrolled * 0.3}px)`;
            heroContent.style.opacity = 1 - (scrollProgress * 0.5);
        }
        
        // Different parallax speeds for floating cells
        floatingCells.forEach((cell, index) => {
            const speed = 0.1 + (index * 0.05); // Varying speeds
            const yPos = scrolled * speed;
            cell.style.transform = `translateY(${yPos}px)`;
        });
        
        // Background overlay effect
        const overlay = heroSection.querySelector('::before');
        if (scrollProgress > 0.5) {
            heroSection.style.setProperty('--overlay-opacity', scrollProgress);
        }
    }, 16); // ~60fps
    
    window.addEventListener('scroll', handleScroll);
    
    // Mouse movement parallax effect
    heroSection.addEventListener('mousemove', function(e) {
        const rect = this.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width - 0.5;
        const y = (e.clientY - rect.top) / rect.height - 0.5;
        
        floatingCells.forEach((cell, index) => {
            const intensity = 10 + (index * 2);
            const moveX = x * intensity;
            const moveY = y * intensity;
            
            cell.style.transform += ` translate(${moveX}px, ${moveY}px)`;
        });
    });
    
    // Reset on mouse leave
    heroSection.addEventListener('mouseleave', function() {
        floatingCells.forEach(cell => {
            cell.style.transform = cell.style.transform.replace(/translate\([^)]*\)/g, '');
        });
    });
}

// ==========================================
// NUMBER ANIMATION
// ==========================================
function animateNumber(element) {
    const finalNumber = parseInt(element.textContent.replace(/[^\d]/g, ''));
    const suffix = element.textContent.replace(/[\d]/g, '');
    const duration = 2000;
    const steps = 60;
    const increment = finalNumber / steps;
    let current = 0;
    
    const timer = setInterval(() => {
        current += increment;
        if (current >= finalNumber) {
            element.textContent = finalNumber + suffix;
            clearInterval(timer);
        } else {
            element.textContent = Math.floor(current) + suffix;
        }
    }, duration / steps);
}