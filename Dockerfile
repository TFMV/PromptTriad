FROM golang:1.22.3

WORKDIR /app

COPY . .

RUN go mod download
RUN go build -o main .

EXPOSE 8080

CMD ["./main"]
