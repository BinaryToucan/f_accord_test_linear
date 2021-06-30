FROM mcr.microsoft.com/dotnet/core/runtime:3.1 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/core/sdk:3.1 AS build
WORKDIR /src
COPY ["F_accord_test_linear/F_accord_test_linear.fsproj", "F_accord_test_linear/"]
RUN dotnet restore "F_accord_test_linear/F_accord_test_linear.fsproj"
COPY . .
WORKDIR "/src/F_accord_test_linear"
RUN dotnet build "F_accord_test_linear.fsproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "F_accord_test_linear.fsproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "F_accord_test_linear.dll"]
