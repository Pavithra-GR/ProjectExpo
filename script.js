document.getElementById('map').addEventListener('click', function() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(function(position) {
            alert('Your current location is: ' + position.coords.latitude + ', ' + position.coords.longitude);
        });
    } else {
        alert('Geolocation is not supported by this browser.');
    }
});
// Script.js
document.querySelector('.edit-avatar').addEventListener('click', function() {
    // Code to handle avatar editing
});

document.querySelector('.save').addEventListener('click', function() {
    // Code to save profile/account information
});

document.querySelector('.verify').addEventListener('click', function() {
    // Code to verify phone number
});

  
