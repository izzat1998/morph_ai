/**
 * HOME PAGE SCRIPTS - MORPH AI
 * Handles animations, interactions, and effects for the home page
 */

document.addEventListener('DOMContentLoaded', function() {
    'use strict';
    
    /**
     * Animated counter function for stats display
     * @param {Element} element - DOM element containing data-target attribute
     */
    function animateCounter(element) {
        const target = parseInt(element.getAttribute('data-target'));
        const duration = 2000; // 2 seconds
        const start = parseInt(element.textContent) || 0;
        const increment = (target - start) / (duration / 16);
        let current = start;
        
        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                element.textContent = target;
                clearInterval(timer);
            } else {
                element.textContent = Math.floor(current);
            }
        }, 16);
    }
    
    /**
     * Intersection Observer for scroll-triggered animations
     */
    const observerOptions = {
        threshold: 0.3,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Animate counters when they come into view
                if (entry.target.classList.contains('hero-stats') || 
                    entry.target.classList.contains('stats-grid') || 
                    entry.target.classList.contains('performance-metrics-new')) {
                    const counters = entry.target.querySelectorAll('[data-target]');
                    counters.forEach(counter => {
                        if (counter.textContent === '0' || counter.textContent === '') {
                            animateCounter(counter);
                        }
                    });
                }
                
                // Add fade-in animations for elements with animate-on-scroll class
                if (entry.target.classList.contains('animate-on-scroll')) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
                
                // Unobserve element to prevent re-triggering
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    /**
     * Set up scroll animations for all relevant elements
     */
    function initializeScrollAnimations() {
        const animatedElements = document.querySelectorAll('.hero-stats, .stats-grid, .performance-metrics-new, .animate-on-scroll');
        
        animatedElements.forEach(el => {
            // Set initial state for animated elements
            if (el.classList.contains('animate-on-scroll')) {
                el.style.opacity = '0';
                el.style.transform = 'translateY(30px)';
                el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            }
            observer.observe(el);
        });
    }
    
    /**
     * Smooth scroll functionality for anchor links
     */
    function initializeSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
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
    }
    
    /**
     * Enhanced feature card interactions
     */
    function initializeFeatureCardInteractions() {
        document.querySelectorAll('.feature-card').forEach(card => {
            card.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-10px) scale(1.02)';
            });
            
            card.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0) scale(1)';
            });
        });
    }
    
    /**
     * Lightweight parallax effect for floating background elements
     */
    function initializeParallaxEffect() {
        let ticking = false;
        
        function updateParallax() {
            const scrolled = window.pageYOffset;
            const parallaxElements = document.querySelectorAll('.floating-cell, .floating-dna, .floating-molecule');
            
            parallaxElements.forEach((element, index) => {
                const speed = 0.5 + (index * 0.1);
                const yPos = -(scrolled * speed);
                const rotation = scrolled * 0.05;
                element.style.transform = `translateY(${yPos}px) rotate(${rotation}deg)`;
            });
            
            ticking = false;
        }
        
        function requestTick() {
            if (!ticking) {
                requestAnimationFrame(updateParallax);
                ticking = true;
            }
        }
        
        window.addEventListener('scroll', requestTick, { passive: true });
    }
    
    /**
     * Enhanced button interactions for custom buttons
     */
    function initializeButtonInteractions() {
        document.querySelectorAll('.btn-primary-custom, .btn-outline-primary-custom').forEach(button => {
            button.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-2px) scale(1.02)';
            });
            
            button.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0) scale(1)';
            });
            
            button.addEventListener('mousedown', function() {
                this.style.transform = 'translateY(0) scale(0.98)';
            });
            
            button.addEventListener('mouseup', function() {
                this.style.transform = 'translateY(-2px) scale(1.02)';
            });
        });
    }
    
    /**
     * Card hover effects for new capability cards
     */
    function initializeCapabilityCardEffects() {
        document.querySelectorAll('.capability-card-new, .stat-card-new, .metric-card-new').forEach(card => {
            card.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-8px) scale(1.01)';
            });
            
            card.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0) scale(1)';
            });
        });
    }
    
    /**
     * Staggered animation delays for capability items
     */
    function initializeStaggeredAnimations() {
        document.querySelectorAll('[data-delay]').forEach(element => {
            const delay = element.getAttribute('data-delay');
            if (delay) {
                element.style.animationDelay = `${delay}ms`;
                element.style.transitionDelay = `${delay}ms`;
            }
        });
    }
    
    /**
     * Performance optimization: Reduce motion for users who prefer it
     */
    function respectUserMotionPreferences() {
        const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
        
        if (prefersReducedMotion.matches) {
            // Disable complex animations for users who prefer reduced motion
            const style = document.createElement('style');
            style.textContent = `
                *, *::before, *::after {
                    animation-duration: 0.01ms !important;
                    animation-iteration-count: 1 !important;
                    transition-duration: 0.01ms !important;
                    scroll-behavior: auto !important;
                }
            `;
            document.head.appendChild(style);
        }
    }
    
    /**
     * Initialize all functionality
     */
    function initialize() {
        initializeScrollAnimations();
        initializeSmoothScroll();
        initializeFeatureCardInteractions();
        initializeParallaxEffect();
        initializeButtonInteractions();
        initializeCapabilityCardEffects();
        initializeStaggeredAnimations();
        respectUserMotionPreferences();
        
        // Log initialization completion
        console.log('🚀 Home page interactions initialized successfully');
    }
    
    // Start initialization
    initialize();
    
    // Expose utility functions globally if needed
    window.MorphAI = window.MorphAI || {};
    window.MorphAI.animateCounter = animateCounter;
});