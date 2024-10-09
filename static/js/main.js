
$(document).ready(function () {
    $("#imageUpload").change(function (event) {
        var reader = new FileReader();
        reader.onload = function (e) {
            $("#selected-image").attr("src", e.target.result);
            $(".image-section").show(); 
            $("#btn-predict").show(); 
            $("#result").hide(); 
            $("#prediction-text").text("");
            $("#btn-predict").prop('disabled', false); 
        };
        reader.readAsDataURL(event.target.files[0]);
    });

    $("#btn-predict").click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        $(this).hide(); 
        $(".loader").show(); 

        $.ajax({
            type: 'POST',
            url: '/predict', 
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                $(".loader").hide(); 
                $("#btn-predict").prop('disabled', true); 
                if (data.prediction) {
                    $("#result").show(); 
                    $("#prediction-text").text(data.prediction); 
                    console.log(data.prediction); 
                } else if (data.error) {
                    $("#result").show();
                    $("#prediction-text").text("Error: " + data.error); 
                }
            },
            error: function (xhr, status, error) {
                $(".loader").hide(); 
                $("#btn-predict").show(); 
                alert("Đã xảy ra lỗi khi dự đoán!");
            }
        });
    });
});
