document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;
    
    // Theme logic
    themeToggle.addEventListener('click', () => {
        if (body.classList.contains('dark-mode')) {
            body.classList.remove('dark-mode');
            body.classList.add('light-mode');
        } else {
            body.classList.remove('light-mode');
            body.classList.add('dark-mode');
        }
    });

    // Auth logic
    const authForm = document.getElementById('auth-form');
    const otpContainer = document.getElementById('otp-container');
    const emailInput = document.getElementById('email');
    const verifyBtn = document.getElementById('verify-btn');
    const otpInput = document.getElementById('otp');
    
    authForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = emailInput.value;
        if(email) {
            // Trigger OTP backend logic here
            console.log(`Sending OTP to ${email}...`);
            // Mock Transition
            authForm.style.display = 'none';
            otpContainer.style.display = 'block';
            otpContainer.style.animation = 'fadeIn 0.5s ease-out';
        }
    });

    verifyBtn.addEventListener('click', () => {
        const o = otpInput.value;
        if(o) {
            console.log(`Verifying OTP: ${o}`);
            // Mock Success Transition
            document.getElementById('auth-section').innerHTML = '<h2>Login Successful!</h2><p>Redirecting to dashboard...</p>';
            setTimeout(() => {
                window.location.href = 'dashboard.html';
            }, 1500);
        }
    });
});
