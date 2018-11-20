

$(document).ready(function(e){
    $("#submitButton").on('click',function(){

        $(this).val("Wait a second...");
        $(this).css('color','red');
    });
});